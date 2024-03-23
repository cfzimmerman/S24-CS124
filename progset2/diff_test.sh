#!/bin/bash

make &&
	cat "./src/test_data/raw_8x8.txt" | head -128 >"temp_in.txt" &&
	cat "./src/test_data/raw_8x8.txt" | tail -64 | awk "NR % 9 == 1" >"temp_out.txt" &&
	./strassen 0 8 "temp_in.txt" >"temp_strassen.txt" &&
	diff "temp_strassen.txt" "temp_out.txt" &&
	echo "✅ Passed"

rm -f "temp_in.txt" "temp_strassen.txt" "temp_out.txt"
