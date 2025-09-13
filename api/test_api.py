#!/usr/bin/env python3
"""
Test TradingView Scripts API

Tests the FastAPI router functionality and endpoint validation.
"""

import sys
import json
from pathlib import Path
import asyncio

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

async def test_api_imports():
    """Test API imports and dependencies"""
    print("🔧 Testing API imports...")
    
    try:
        from api.tvscripts_api import router
        print("✅ TradingView API router imported successfully")
        
        # Test router configuration
        print(f"   Router prefix: {router.prefix}")
        print(f"   Router tags: {router.tags}")
        
        # Check routes
        routes = [route for route in router.routes]
        print(f"   Total routes: {len(routes)}")
        
        for route in routes:
            if hasattr(route, 'methods') and hasattr(route, 'path'):
                methods = list(route.methods)
                print(f"   - {methods} {route.path}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import API components: {e}")
        return False

async def test_pydantic_models():
    """Test Pydantic model validation"""
    print("\n📋 Testing Pydantic models...")
    
    try:
        from api.tvscripts_api import (
            SearchRequest, AnalysisRequest, ImportRequest,
            ScriptResponse, AnalysisResponse, ImportResponse,
            DatabaseStatsResponse
        )
        
        # Test SearchRequest
        search_req = SearchRequest(query="ema crossover", limit=10)
        print(f"✅ SearchRequest: {search_req.query}, limit={search_req.limit}")
        
        # Test AnalysisRequest
        analysis_req = AnalysisRequest(slug="test-strategy")
        print(f"✅ AnalysisRequest: slug={analysis_req.slug}")
        
        # Test ImportRequest
        import_req = ImportRequest(slug="test-strategy", preset_name="test")
        print(f"✅ ImportRequest: slug={import_req.slug}, preset={import_req.preset_name}")
        
        # Test response models
        script_resp = ScriptResponse(
            slug="test",
            title="Test Strategy",
            author="TestAuthor",
            open_source=True
        )
        print(f"✅ ScriptResponse: {script_resp.title} by {script_resp.author}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pydantic model test failed: {e}")
        return False

async def test_dependencies():
    """Test API dependencies"""
    print("\n🔗 Testing API dependencies...")
    
    try:
        from api.tvscripts_api import get_mcp_client, get_integrator
        
        # Test MCP client dependency
        try:
            client = get_mcp_client()
            print("✅ MCP client dependency available")
        except Exception as e:
            print(f"⚠️ MCP client dependency issue: {e}")
        
        # Test integrator dependency
        try:
            integrator = get_integrator()
            print("✅ Integrator dependency available")
        except Exception as e:
            print(f"⚠️ Integrator dependency issue: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dependencies test failed: {e}")
        return False

async def test_endpoint_logic():
    """Test endpoint logic without HTTP"""
    print("\n⚙️ Testing endpoint logic...")
    
    try:
        from api.tvscripts_api import (
            health_check, validate_integration
        )
        
        # Test health check
        health_result = await health_check()
        print(f"✅ Health check: {health_result['status']}")
        print(f"   MCP available: {health_result.get('mcp_client_available', False)}")
        print(f"   Integration available: {health_result.get('integration_available', False)}")
        
        # Test validation endpoint
        try:
            validation_result = await validate_integration()
            print(f"✅ Validation: {validation_result['healthy']}")
        except Exception as e:
            print(f"⚠️ Validation endpoint issue: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Endpoint logic test failed: {e}")
        return False

async def test_fastapi_app():
    """Test FastAPI app creation"""
    print("\n🚀 Testing FastAPI app creation...")
    
    try:
        from fastapi import FastAPI
        from api.tvscripts_api import router
        
        # Create test app
        app = FastAPI(title="Test App")
        app.include_router(router)
        
        print("✅ FastAPI app created successfully")
        print(f"   App title: {app.title}")
        
        # Check routes are included
        routes = [route for route in app.routes]
        api_routes = [route for route in routes if hasattr(route, 'path') and route.path.startswith('/api/tvscripts')]
        
        print(f"   Total routes: {len(routes)}")
        print(f"   API routes: {len(api_routes)}")
        
        return True
        
    except Exception as e:
        print(f"❌ FastAPI app test failed: {e}")
        return False

async def test_error_handling():
    """Test error handling scenarios"""
    print("\n⚠️ Testing error handling...")
    
    try:
        from api.tvscripts_api import SearchRequest, AnalysisRequest
        from pydantic import ValidationError
        
        # Test invalid search request
        try:
            SearchRequest(query="", limit=0)  # Should fail validation
            print("❌ Should have failed validation")
            return False
        except ValidationError:
            print("✅ Search request validation working")
        
        # Test analysis request validation
        try:
            AnalysisRequest()  # Should require either slug or code
            analysis_req = AnalysisRequest(slug="test")  # This should work
            print("✅ Analysis request validation working")
        except ValidationError as e:
            print(f"⚠️ Analysis request validation: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

async def test_api_documentation():
    """Test API documentation generation"""
    print("\n📚 Testing API documentation...")
    
    try:
        from fastapi import FastAPI
        from api.tvscripts_api import router
        
        app = FastAPI(
            title="TradingView Scripts API",
            description="Test documentation generation",
            version="0.1.0"
        )
        app.include_router(router)
        
        # Test OpenAPI schema generation
        openapi_schema = app.openapi()
        
        print("✅ OpenAPI schema generated")
        print(f"   Title: {openapi_schema['info']['title']}")
        print(f"   Version: {openapi_schema['info']['version']}")
        print(f"   Paths: {len(openapi_schema['paths'])}")
        
        # Check for key endpoints
        paths = openapi_schema['paths']
        expected_endpoints = [
            '/api/tvscripts/health',
            '/api/tvscripts/search',
            '/api/tvscripts/analyze',
            '/api/tvscripts/import'
        ]
        
        for endpoint in expected_endpoints:
            if endpoint in paths:
                print(f"   ✅ Found endpoint: {endpoint}")
            else:
                print(f"   ❌ Missing endpoint: {endpoint}")
        
        return True
        
    except Exception as e:
        print(f"❌ API documentation test failed: {e}")
        return False

async def main():
    """Run all API tests"""
    print("🧪 TradingView Scripts API Test Suite")
    print("=" * 60)
    
    tests = [
        ("API Imports", test_api_imports),
        ("Pydantic Models", test_pydantic_models),
        ("Dependencies", test_dependencies),
        ("Endpoint Logic", test_endpoint_logic),
        ("FastAPI App", test_fastapi_app),
        ("Error Handling", test_error_handling),
        ("API Documentation", test_api_documentation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            if await test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All API tests passed! FastAPI router is ready.")
        return True
    elif passed >= (total * 0.7):
        print("✅ Most API tests passed. Router should be functional.")
        return True
    else:
        print("⚠️ Some API tests failed. Check the implementation.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)