DIRS = C OpenCL

all:
	set -e; for d in $(DIRS); do $(MAKE) -C $$d ; done

clobber:
	set -e; for d in $(DIRS); do $(MAKE) -C $$d clobber; done

clean:
	set -e; for d in $(DIRS); do $(MAKE) -C $$d clean; done
