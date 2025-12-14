#!/bin/bash

cd "$(dirname "$0")"

echo "Building GokiRunMetal..."
swiftc -O -parse-as-library -target arm64-apple-macosx15.0 GokiRunMetal.swift -o GokiRunMetal 2>&1

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Run with: ./GokiRunMetal [-c COUNT]"
else
    echo "Build failed!"
    exit 1
fi
