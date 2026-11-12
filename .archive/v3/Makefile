CC = gcc
# CFLAGS = -Wall -Wextra -O2
CFLAGS = 

main: main.c autograd.c autograd.h
	$(CC) $(CFLAGS) main.c autograd.c -o $@ -lm

test_autograd: test_autograd.c autograd.c autograd.h
	$(CC) $(CFLAGS) test_autograd.c autograd.c -o $@ -lm

timing: time.c autograd.c autograd.h
	$(CC) $(CFLAGS) time.c autograd.c -o $@ -lm

test: test_autograd
	./test_autograd

run: main
	./main

time: timing
	./timing

clean:
	rm -f main test_autograd

.PHONY: run clean
