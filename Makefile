SHELL := /bin/bash

SUBDIRS := kernels

.PHONY: all clean dumps subdirs $(SUBDIRS)

all: dumps

subdirs: $(SUBDIRS)

$(SUBDIRS):
	@$(MAKE) -C $@

dumps: subdirs
	@for dir in $(SUBDIRS); do \
	  if [ -d "$$dir/build" ]; then \
	    while IFS= read -r bin; do \
	      base=$${bin%.*}; \
	      ptx=$${base}.ptx; \
	      sass=$${base}.sass; \
	      if [ ! -f "$$ptx" ] || [ "$$bin" -nt "$$ptx" ]; then \
	        echo "Generating $$ptx"; \
	        cuobjdump --dump-ptx "$$bin" > "$$ptx"; \
	      fi; \
	      if [ ! -f "$$sass" ] || [ "$$bin" -nt "$$sass" ]; then \
	        echo "Generating $$sass"; \
	        cuobjdump --dump-sass "$$bin" > "$$sass"; \
	      fi; \
	    done < <(find "$$dir/build" -maxdepth 1 -type f \( -name '*.cubin' -o -name '*.out' \)); \
	  fi; \
	done

clean:
	@for dir in $(SUBDIRS); do \
	  $(MAKE) -C $$dir clean; \
	  if [ -d "$$dir/build" ]; then \
	    find "$$dir/build" -maxdepth 1 -type f \( -name '*.ptx' -o -name '*.sass' \) -delete; \
	  fi; \
	done
