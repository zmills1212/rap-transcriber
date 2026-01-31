"""
Test script for the Rap Transcription API.
Tests all endpoints without requiring a running server.
"""
import sys
sys.path.insert(0, '.')

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


def test_server_imports():
    """Test that server modules import correctly."""
    print("Testing server imports...")
    
    from src.api.server import app
    from src.api.client import TranscriptionClient
    from src.api.websocket import ConnectionManager, add_websocket_routes
    
    print("   ✅ server.py imported")
    print("   ✅ client.py imported")
    print("   ✅ websocket.py imported")
    
    return True


def test_fastapi_app():
    """Test FastAPI app configuration."""
    print("Testing FastAPI app...")
    
    from src.api.server import app
    
    # Check app exists
    assert app is not None
    assert app.title == "Rap Transcription API"
    print(f"   ✅ App title: {app.title}")
    print(f"   ✅ App version: {app.version}")
    
    # Check routes exist
    routes = [route.path for route in app.routes]
    
    expected_routes = ['/', '/health', '/transcribe', '/format']
    for route in expected_routes:
        assert route in routes, f"Missing route: {route}"
        print(f"   ✅ Route exists: {route}")
    
    return True


def test_pydantic_models():
    """Test Pydantic request/response models."""
    print("Testing Pydantic models...")
    
    from src.api.server import (
        TranscriptionResponse,
        TranscriptionRequest,
        JobStatus,
        HealthResponse
    )
    
    # Test TranscriptionResponse
    response = TranscriptionResponse(
        text="test transcription",
        processing_time_seconds=1.5
    )
    assert response.text == "test transcription"
    print("   ✅ TranscriptionResponse")
    
    # Test TranscriptionRequest
    request = TranscriptionRequest(
        text="test text",
        format_as_lyrics=True
    )
    assert request.format_as_lyrics == True
    print("   ✅ TranscriptionRequest")
    
    # Test JobStatus
    job = JobStatus(
        job_id="123",
        status="pending"
    )
    assert job.status == "pending"
    print("   ✅ JobStatus")
    
    # Test HealthResponse
    health = HealthResponse(
        status="healthy",
        model_loaded=False
    )
    assert health.status == "healthy"
    print("   ✅ HealthResponse")
    
    return True


def test_client_class():
    """Test API client class."""
    print("Testing API client...")
    
    from src.api.client import TranscriptionClient
    
    # Test initialization
    client = TranscriptionClient("http://localhost:8000")
    assert client.base_url == "http://localhost:8000"
    print("   ✅ Client initialization")
    
    # Test URL handling
    client2 = TranscriptionClient("http://localhost:8000/")
    assert client2.base_url == "http://localhost:8000"
    print("   ✅ URL trailing slash handling")
    
    # Test methods exist
    assert hasattr(client, 'health_check')
    assert hasattr(client, 'transcribe')
    assert hasattr(client, 'transcribe_async')
    assert hasattr(client, 'format_text')
    print("   ✅ All client methods exist")
    
    return True


def test_websocket_manager():
    """Test WebSocket connection manager."""
    print("Testing WebSocket manager...")
    
    from src.api.websocket import ConnectionManager
    
    manager = ConnectionManager()
    
    assert manager.active_connections == []
    print("   ✅ ConnectionManager initialized")
    print(f"   ✅ Active connections: {len(manager.active_connections)}")
    
    return True


def test_endpoint_handlers():
    """Test endpoint handler functions exist."""
    print("Testing endpoint handlers...")
    
    from src.api.server import (
        root,
        health_check,
        transcribe,
        format_text
    )
    
    import asyncio
    
    # Test root endpoint
    result = asyncio.get_event_loop().run_until_complete(root())
    assert 'name' in result
    assert result['name'] == "Rap Transcription API"
    print("   ✅ root() handler")
    
    # Test health endpoint
    result = asyncio.get_event_loop().run_until_complete(health_check())
    assert result.status == "healthy"
    print("   ✅ health_check() handler")
    
    return True


def test_with_test_client():
    """Test API using FastAPI TestClient."""
    print("Testing with TestClient...")
    
    try:
        from fastapi.testclient import TestClient
        from src.api.server import app
        
        client = TestClient(app)
        
        # Test root
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data['name'] == "Rap Transcription API"
        print("   ✅ GET / returns 200")
        
        # Test health
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == "healthy"
        print("   ✅ GET /health returns 200")
        
        # Test format endpoint
        response = client.post("/format", json={
            "text": "i'm finna get bread",
            "format_as_lyrics": False,
            "normalize_slang": False
        })
        assert response.status_code == 200
        data = response.json()
        assert 'formatted' in data
        print("   ✅ POST /format returns 200")
        
        return True
        
    except ImportError:
        print("   ⚠️ httpx not installed, skipping TestClient tests")
        print("   Run: pip install httpx")
        return True


def main():
    print("=" * 50)
    print("RAP TRANSCRIPTION API TEST")
    print("=" * 50)
    print("")
    
    tests = [
        ("Server Imports", test_server_imports),
        ("FastAPI App", test_fastapi_app),
        ("Pydantic Models", test_pydantic_models),
        ("API Client", test_client_class),
        ("WebSocket Manager", test_websocket_manager),
        ("Endpoint Handlers", test_endpoint_handlers),
        ("TestClient Integration", test_with_test_client),
    ]
    
    results = []
    
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
        print("")
    
    # Summary
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"   {status} {name}")
    
    print("")
    print(f"   Passed: {passed}/{total}")
    
    if passed == total:
        print("")
        print("🎉 All tests passed! API is ready.")
        print("")
        print("To start the server:")
        print("  python scripts/run_api.py")
        print("")
        print("Then visit: http://localhost:8000/docs")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
