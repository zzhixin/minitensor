CC = gcc
# CFLAGS = -Wall -Wextra -O2
CFLAGS = 

main: main.c autograd.c autograd.h
	$(CC) $(CFLAGS) main.c autograd.c -o $@ -lm

run: main
	./main

clean:
	rm -f main

.PHONY: run clean
