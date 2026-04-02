"""
Genius Lyrics Fetcher
======================
Searches for songs via the Genius API and scrapes lyrics from the page.

The Genius API provides search and song metadata, but not lyrics directly.
Lyrics are scraped from the song's Genius webpage using BeautifulSoup.

Usage:
    from lyrics_matcher.genius import GeniusClient

    genius = GeniusClient()  # reads GENIUS_API_TOKEN from .env
    lyrics = genius.fetch_lyrics("Niggas Be Lame", artist="Yung Bans")
    print(lyrics)

CLI:
    python -m lyrics_matcher.genius "Niggas Be Lame" --artist "Yung Bans"
"""

import os
import re
import requests
from pathlib import Path
from typing import Optional
from bs4 import BeautifulSoup


def _load_token() -> str:
    """Load Genius API token from .env file or environment."""
    # Check environment first
    token = os.environ.get("GENIUS_API_TOKEN")
    if token:
        return token

    # Try .env file in project root
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("GENIUS_API_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")

    raise ValueError(
        "GENIUS_API_TOKEN not found. Set it in .env or as an environment variable.\n"
        "Get one at: https://genius.com/api-clients"
    )


class GeniusClient:
    """Fetch lyrics from Genius."""

    BASE_URL = "https://api.genius.com"

    def __init__(self, token: Optional[str] = None):
        self.token = token or _load_token()
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.token}",
            "User-Agent": "RapTranscriber/1.0",
        })

    def search(self, query: str, limit: int = 5) -> list:
        """
        Search Genius for songs matching a query.
        Returns list of dicts with: title, artist, url, id
        """
        resp = self.session.get(
            f"{self.BASE_URL}/search",
            params={"q": query, "per_page": limit},
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        for hit in data.get("response", {}).get("hits", []):
            song = hit.get("result", {})
            results.append({
                "title": song.get("title", ""),
                "artist": song.get("primary_artist", {}).get("name", ""),
                "url": song.get("url", ""),
                "id": song.get("id"),
            })

        return results

    def fetch_lyrics(
        self,
        song_title: str,
        artist: Optional[str] = None,
    ) -> Optional[str]:
        """
        Search for a song and return its lyrics.

        Args:
            song_title: Song name to search for
            artist: Optional artist name to improve search accuracy

        Returns:
            Lyrics as plain text, or None if not found
        """
        # Build search query
        query = f"{artist} {song_title}" if artist else song_title
        results = self.search(query)

        if not results:
            print(f"  No results found for: {query}")
            return None

        # Find best match
        best = self._pick_best_match(results, song_title, artist)
        if not best:
            print(f"  No matching song found for: {query}")
            print(f"  Top results:")
            for r in results[:3]:
                print(f"    - {r['artist']} - {r['title']}")
            return None

        print(f"  Found: {best['artist']} - {best['title']}")

        # Scrape lyrics from the Genius page
        lyrics = self._scrape_lyrics(best["url"])
        return lyrics

    def _pick_best_match(
        self, results: list, title: str, artist: Optional[str]
    ) -> Optional[dict]:
        """Pick the best matching result based on title and artist similarity."""
        title_lower = title.lower().strip()
        artist_lower = (artist or "").lower().strip()

        for result in results:
            r_title = result["title"].lower().strip()
            r_artist = result["artist"].lower().strip()

            # Exact title match (with optional artist match)
            if r_title == title_lower:
                if not artist_lower or artist_lower in r_artist or r_artist in artist_lower:
                    return result

        # Fuzzy: title contained in result or vice versa
        for result in results:
            r_title = result["title"].lower().strip()
            r_artist = result["artist"].lower().strip()

            title_match = title_lower in r_title or r_title in title_lower
            artist_match = (
                not artist_lower
                or artist_lower in r_artist
                or r_artist in artist_lower
            )

            if title_match and artist_match:
                return result

        # If artist specified but no match, try first result with matching artist
        if artist_lower:
            for result in results:
                if artist_lower in result["artist"].lower():
                    return result

        # Last resort: return first result
        return results[0] if results else None

    def _scrape_lyrics(self, url: str) -> Optional[str]:
        """Scrape lyrics text from a Genius song page."""
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": "RapTranscriber/1.0"},
                timeout=10,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  Error fetching page: {e}")
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        # Genius wraps lyrics in divs with data-lyrics-container="true"
        lyrics_containers = soup.find_all("div", attrs={"data-lyrics-container": "true"})

        if not lyrics_containers:
            print(f"  Could not find lyrics on page: {url}")
            return None

        lines = []
        for container in lyrics_containers:
            # Replace <br> with newlines before extracting text
            for br in container.find_all("br"):
                br.replace_with("\n")
            text = container.get_text()
            lines.append(text)

        raw_lyrics = "\n".join(lines)

        # Clean up
        cleaned = self._clean_lyrics(raw_lyrics)
        return cleaned

    def _clean_lyrics(self, text: str) -> str:
        """Clean scraped lyrics text."""
        # Remove section headers like [Chorus], [Verse 1], [Intro]
        text = re.sub(r"\[.*?\]", "", text)

        # Remove Genius metadata from first line
        text = re.sub(r"^\d+\s*Contributor[s]?", "", text)
        text = re.sub(r"^.*?Lyrics", "", text, count=1)
        text = re.sub(r'["“][^"”]*["”]\s+is a track by.*?Read More', "", text, flags=re.DOTALL)
        text = re.sub(r'.*?is a (track|song) by.*?Read More', "", text, flags=re.DOTALL)

        # Remove contributor/header lines
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Skip empty lines and metadata
            if not line:
                continue
            if line.endswith("Lyrics"):
                continue
            if line.startswith("See ") and "live" in line.lower():
                continue
            if "Embed" in line:
                continue
            if line.endswith("Contributors"):
                continue
            if re.match(r"^\d+$", line):  # standalone numbers
                continue
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fetch lyrics from Genius")
    parser.add_argument("song", help="Song title to search for")
    parser.add_argument("--artist", "-a", help="Artist name")
    parser.add_argument("--raw", action="store_true", help="Show raw (uncleaned) lyrics")

    args = parser.parse_args()

    genius = GeniusClient()

    print(f"\nSearching Genius for: {args.song}")
    if args.artist:
        print(f"  Artist: {args.artist}")

    lyrics = genius.fetch_lyrics(args.song, artist=args.artist)

    if lyrics:
        print(f"\n{'=' * 60}")
        print(f"LYRICS")
        print(f"{'=' * 60}\n")
        print(lyrics)
    else:
        print("\nNo lyrics found.")


if __name__ == "__main__":
    main()
