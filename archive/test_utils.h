// test_utils.h
// Simple lightweight assertion helpers for unit tests.
// No external dependency. Prints pass/fail with context.

#pragma once
#include <cmath>
#include <iostream>

inline void expect_close(const char* name, double a, double b, double tol = 1e-3) {
    if (std::abs(a - b) <= tol) {
        std::cout << "[PASS] " << name << ": " << a << " ≈ " << b << "\n";
    } else {
        std::cout << "[FAIL] " << name << ": " << a << " != " << b << " (expected " << b << ")\n";
    }
}

inline void expect_true(const char* name, bool cond) {
    if (cond) {
        std::cout << "[PASS] " << name << "\n";
    } else {
        std::cout << "[FAIL] " << name << " is false\n";
    }
}

inline void expect_eq_str(const char* name, const std::string& a, const std::string& b) {
    if (a == b) {
        std::cout << "[PASS] " << name << ": \"" << a << "\" == \"" << b << "\"\n";
    } else {
        std::cout << "[FAIL] " << name << ": \"" << a << "\" != \"" << b << "\"\n";
    }
}
