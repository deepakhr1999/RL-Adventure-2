sudo fuser -v /dev/nvidia0 2>&1 | grep python | grep -o -E " [0-9]+ " | xargs kill
