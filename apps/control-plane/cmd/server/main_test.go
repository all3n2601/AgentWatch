package main

import (
	"net/http/httptest"
	"strings"
	"testing"
)

func TestHealth(t *testing.T) {
	request := httptest.NewRequest("GET", "/healthz", nil)
	response := httptest.NewRecorder()

	health(response, request)

	if response.Code != 200 {
		t.Fatalf("expected status 200, got %d", response.Code)
	}
	if !strings.Contains(response.Body.String(), `"status":"ok"`) {
		t.Fatalf("unexpected response: %s", response.Body.String())
	}
}
