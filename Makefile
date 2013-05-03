CC := nvcc
CFLAGS := -g -G -O0 -arch=sm_10
LDFLAGS := -L/opt/cuda/lib64/ -lcuda -lcudart

.PHONY: all clean

EXE := ann_cuda
OBJ := main.o data.o backprop.o

all: $(EXE)
	
$(EXE): $(OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -o $@ -c $<

%.o: %.cu
	$(CC) $(CFLAGS) -o $@ -c $<

clean:
	rm -f *.o $(EXE)

