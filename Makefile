CC = gcc
CFLAGS = -Wall -Wextra -O0
# CFLAGS = 

main: main.c autograd.c tensor.c autograd.h tensor.h
	$(CC) $(CFLAGS) main.c autograd.c tensor.c -o $@ -lm

test_autograd: test_autograd.c autograd.c  autograd.h random.h
	$(CC) $(CFLAGS) test_autograd.c autograd.c  -o $@ -lm

test_tensor: test_tensor.c tensor.c  tensor.h random.h
	$(CC) $(CFLAGS) test_tensor.c tensor.c  -o $@ -lm

timing: time.c autograd.c  autograd.h random.h
	$(CC) $(CFLAGS) time.c autograd.c  -o $@ -lm

test: test_autograd test_tensor
	./test_autograd && ./test_tensor

run: main
	./main

time: timing
	./timing

clean:
	rm -f main test_autograd

.PHONY: run clean
