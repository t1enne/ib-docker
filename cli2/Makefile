PROJECT = IbkrCli
JULIA = julia --project=. 


run:
	$(JULIA) -e "import IbkrCli; IbkrCli.main()" $(ARGS)

# Run tests (once you add a test/ folder)
test:
	$(JULIA) -e 'using Pkg; Pkg.test("$(PROJECT)")'

# Install this package into Julia's environment
install:
	$(JULIA) -e 'using Pkg; Pkg.develop(path=".")'

# Remove the package from Julia's environment
uninstall:
	$(JULIA) -e 'using Pkg; Pkg.rm("$(PROJECT)")'

# Clean up build artifacts (if you generate anything)
clean:
	rm -rf Manifest.toml .julia

.PHONY: default run test install uninstall clean
# forward args
%:
	@:
