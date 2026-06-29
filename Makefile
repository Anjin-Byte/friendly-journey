# friendly-journey — developer command catalog.
#
# A thin convenience layer over the canonical automation: CI logic lives in
# `cargo xtask` (Engineering Codex: build automation is a workspace member under
# tools/), the headless tools in `voxel-cli`, the viewer in `voxel-viewer`. This
# Makefile only ALIASES those — it holds no logic of its own, so `cargo xtask ci`
# stays the single source of truth for the gate.
#
# Parameters override on the command line, e.g. `make capture RES=2048 FIXTURE=dust`.
# Run `make` (or `make help`) for the list.

FIXTURE ?= dust
RES     ?= 512
ITERS   ?= 5000
RAYS    ?= 20000
MESH    ?= models/gltf/LittlestTokyo.glb
# Corrective X rotation for the default OBJ (authored Z-up; this viewer is Y-up).
ROT_X   ?= -90

CLI := cargo run --release -p voxel-cli --
BIN := ./target/release/voxel

.DEFAULT_GOAL := help

.PHONY: help
help: ## list available targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

## --- quality gate ---

.PHONY: ci
ci: ## full gate: fmt-check, clippy -D warnings, build, test, doc
	cargo xtask ci

.PHONY: ci-gpu
ci-gpu: ## CI plus the GPU-vs-mirror differential (needs an adapter)
	cargo xtask ci-gpu

.PHONY: fmt
fmt: ## format the workspace
	cargo fmt --all

.PHONY: test
test: ## run the test suite
	cargo test --workspace

.PHONY: build
build: ## release-build the whole workspace
	cargo build --release --workspace

.PHONY: doc
doc: ## build + open the docs
	cargo doc --workspace --no-deps --open

## --- interactive ---

.PHONY: viewer
viewer: ## run the viewer (FIXTURE, RES); right-click edits, B / [ ] brush
	cargo run --release -p voxel-viewer -- --fixture $(FIXTURE) --res $(RES)

.PHONY: mesh
mesh: ## voxelize + view a mesh file (MESH, RES, ROT_X; gltf/glb/obj/stl)
	cargo run --release -p voxel-viewer -- --mesh $(MESH) --res $(RES) --truecolor

## --- headless measurements (FIXTURE, RES) ---

.PHONY: measure
measure: ## §10 report: dimension D, footprint, descent frequency
	$(CLI) measure --fixture $(FIXTURE) --res $(RES)

.PHONY: bench
bench: ## fixtures × resolutions build/size/throughput table
	$(CLI) bench

.PHONY: aniso
aniso: ## orientation cost sweep (algorithmic vs hardware)
	$(CLI) aniso --fixture $(FIXTURE) --res $(RES)

.PHONY: diff
diff: ## GPU-vs-f64-reference hit differential (RAYS)
	$(CLI) diff --backend auto --fixture $(FIXTURE) --res $(RES) --rays $(RAYS)

.PHONY: edit
edit: ## edit-performance suite; add SWEEP=1 for the fixture×res table
	$(CLI) edit --fixture $(FIXTURE) --res $(RES) $(if $(SWEEP),--sweep,)

## --- GPU profiling (see CAPTURE.md) ---

.PHONY: capture
capture: ## loop the worst-orientation kernel as a profiler target (ITERS)
	$(CLI) capture --fixture $(FIXTURE) --res $(RES) --iters $(ITERS)

.PHONY: gputrace
gputrace: build ## write a .gputrace at the worst orientation and open it in Xcode (FIXTURE, RES)
	METAL_CAPTURE_ENABLED=1 $(BIN) capture --fixture $(FIXTURE) --res $(RES) --gputrace
	open ./$(FIXTURE)-$(RES).gputrace

.PHONY: trace
trace: build ## record a Metal System Trace of the capture target → .trace (needs full Xcode)
	@xcrun -f xctrace >/dev/null 2>&1 || { \
	  echo "xctrace not found — point xcrun at full Xcode, then retry:"; \
	  echo "  sudo xcode-select -s /Applications/Xcode.app/Contents/Developer"; \
	  exit 1; }
	xcrun xctrace record --template 'Metal System Trace' \
	  --output ./$(FIXTURE)-$(RES).trace --launch -- \
	  $(BIN) capture --fixture $(FIXTURE) --res $(RES) --iters $(ITERS)
	open ./$(FIXTURE)-$(RES).trace

.PHONY: clean
clean: ## remove build artifacts
	cargo clean
