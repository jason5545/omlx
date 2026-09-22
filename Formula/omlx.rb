class Omlx < Formula
  CUSTOM_KERNELS = %w[bonsai decode_fast glm_moe_dsa minimax_m3 qwen35_prefill].freeze

  desc "LLM inference server optimized for Apple Silicon"
  homepage "https://github.com/jason5545/omlx"
  url "https://github.com/jundot/omlx/archive/refs/tags/v0.6.4.tar.gz"
  sha256 "5d8781c0c6a782e9b90f071d36b0d8dbe6161fafb5fef5e3f0d8ac71da214b70"
  license "Apache-2.0"

  head "https://github.com/jason5545/omlx.git", branch: "main"

  option "with-custom-kernel",
         "Build native custom kernels for Bonsai, GLM-5.2, MiniMax M3 and Qwen3.5/3.6/4 acceleration"
  option "with-grammar", "Install xgrammar for structured output (requires torch, ~2GB)"

  depends_on "rust" => :build
  depends_on arch: :arm64
  depends_on macos: :sequoia
  depends_on "python@3.11"

  # macOS 27 beta's `strip` corrupts dynamic offsets in Mach-O libraries
  # (llvm/llvm-project#203678). Skip Homebrew's post-install clean pass over
  # the venv so it never runs `strip` on the compiled dylibs.
  on_macos do
    skip_clean "libexec" if MacOS.version >= "27"
  end

  # mlx-audio pins mlx-lm==0.31.1 which conflicts with omlx's git-pinned
  # mlx-lm. Fetch source separately so we can patch the pin before install.
  resource "mlx-audio" do
    url "https://github.com/Blaizzy/mlx-audio.git",
      revision: "51753266e0a4f766fd5e6fbc46652224efc23981"
  end

  # Kokoro's English G2P path uses misaki + spaCy. Bundle the spaCy
  # language model so the first TTS request does not download into the
  # Homebrew venv at runtime.
  resource "en-core-web-sm" do
    url "https://github.com/explosion/spacy-models/releases/download/" \
        "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
    sha256 "1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85"
  end

  service do
    run [opt_bin/"omlx", "serve"]
    keep_alive true
    working_dir var
    log_path var/"log/omlx.log"
    error_log_path var/"log/omlx.log"
    environment_variables PATH: std_service_path_env
  end

  def install
    # Create venv with pip so dependency resolution works properly
    system "python3.11", "-m", "venv", libexec

    # Build native extensions from source with headerpad so Homebrew can
    # rewrite Mach-O install names to absolute Cellar/opt paths. Rust/maturin
    # extension builds (cohere_melody) need the linker flag via RUSTFLAGS;
    # C/C++ extension builds use LDFLAGS.
    ENV.append "LDFLAGS", "-Wl,-headerpad_max_install_names"
    ENV.append "RUSTFLAGS", "-C link-arg=-Wl,-headerpad_max_install_names"

    no_binary = "cohere_melody,pydantic-core,rpds-py,tiktoken"
    pip_flags = []
    if MacOS.version >= "27"
      # macOS 27's dyld requires the LC_SYMTAB string pool to start on an
      # 8-byte boundary; prebuilt Rust wheels aligned to 4 bytes fail dlopen
      # with "mis-aligned LINKEDIT string pool". Build them from source, and
      # keep Cargo/maturin's release stripping off so the beta's broken
      # `strip` (llvm/llvm-project#203678) never touches the fresh dylibs.
      no_binary += ",tokenizers"
      ENV["CARGO_PROFILE_RELEASE_STRIP"] = "false"
      ENV["MATURIN_STRIP"] = "false"
      # Pip reuses locally built wheels even under --no-binary, so a wheel
      # cached before the strip guards existed stays corrupted. Bypass the
      # cache entirely.
      pip_flags << "--no-cache-dir"
    end

    # Every pip step must share these flags; a later step without --no-binary
    # (e.g. mlx-audio) can clobber a source-built package with a prebuilt
    # wheel that fails dlopen on macOS 27 (#2110).
    pip_install = [libexec/"bin/pip", "install", *pip_flags, "--no-binary", no_binary]

    if build.with?("custom-kernel")
      kernel_sources = CUSTOM_KERNELS.map do |kernel|
        buildpath/"omlx/custom_kernels/#{kernel}/csrc"
      end
      unless kernel_sources.all?(&:directory?)
        odie "--with-custom-kernel requires oMLX custom kernel sources; use --HEAD or a release that includes them"
      end

      ENV["OMLX_WITH_CUSTOM_KERNEL"] = "1"
      # Pin CMake to the venv's Python; its default discovery can pick a
      # newer unlinked system Python instead. setup.py forwards CMAKE_ARGS
      # to the kernel builds.
      ENV.append "CMAKE_ARGS", "-DPython_EXECUTABLE=#{libexec}/bin/python"
    end

    # Install omlx (with optional grammar extra for structured output)
    install_spec = build.with?("grammar") ? "#{buildpath}[grammar]" : buildpath.to_s
    system(*pip_install, install_spec)

    if build.with?("custom-kernel")
      # Run from libexec so buildpath's raw omlx/ source tree doesn't shadow
      # the compiled package in the venv's site-packages.
      Dir.chdir(libexec) do
        verify_custom_kernels(libexec/"bin/python")
      end
    end

    # Install mlx-audio with patched mlx-lm pin to avoid version conflict
    resource("mlx-audio").stage do
      inreplace "pyproject.toml", '"mlx-lm==0.31.1"', '"mlx-lm>=0.31.1"'
      system(*pip_install, ".[all]")
    end

    # Install the spaCy English model required by misaki for Kokoro TTS.
    # Homebrew's cached resource path is hash-prefixed, which pip rejects
    # as an invalid wheel filename. Copy it back to the canonical basename.
    spacy_model_wheel = buildpath/"en_core_web_sm-3.8.0-py3-none-any.whl"
    cp resource("en-core-web-sm").cached_download, spacy_model_wheel
    system libexec/"bin/pip", "install", "--no-deps",
           spacy_model_wheel
    system libexec/"bin/python", "-c",
           "import spacy; spacy.load('en_core_web_sm')"

    # python-multipart is declared in omlx's [audio] extra, not in mlx-audio
    system(*pip_install, "python-multipart>=0.0.5")

    # jang-tools serves JANG mixed-precision bundles; omlx routes them via
    # omlx.patches.jang_load. Installed here rather than through omlx's own
    # dependency list because the release predates the DMG layer's
    # exclude-newer cutoff in packaging/venvstacks.toml. Every requirement
    # (mlx, mlx-lm, safetensors, numpy, tqdm, huggingface_hub, jinja2) is
    # already pinned above, so resolution only adds the pure-Python package.
    system(*pip_install, "jang[mlx]>=2.5.47")

    bin.install_symlink Dir[libexec/"bin/omlx"]
  end

  # Both fixups below must run after Homebrew's post-install "Cleaning" step,
  # which rewrites Mach-O install names and removes dist-info/RECORD files.
  # Keep this declarative so Homebrew can run it from the formula's JSON API.
  post_install_steps do
    run "/bin/sh", args: ["-euc", <<~SH], writable_paths: ["."], writable_base: :prefix
      python="{{prefix}}/libexec/bin/python"
      site="$($python -c 'import site; print(site.getsitepackages()[0])')"

      # xgrammar's arm64 wheel omits the tvm_ffi rpath and RECORD entry.
      dylib="$site/xgrammar/libxgrammar_bindings.dylib"
      if [ -f "$dylib" ]; then
        tvmlib="$($python -c 'import os, tvm_ffi; print(os.path.join(os.path.dirname(tvm_ffi.__file__), "lib"))')"
        if ! /usr/bin/otool -l "$dylib" | /usr/bin/grep -Fq "$tvmlib"; then
          /usr/bin/install_name_tool -add_rpath "$tvmlib" "$dylib"
          /usr/bin/codesign --force --sign - "$dylib"
        fi
        dist_dir=$(/usr/bin/find "$site" -maxdepth 1 -type d -name 'xgrammar-*.dist-info' -print -quit)
        test -n "$dist_dir"
        record="$dist_dir/RECORD"
        if ! /usr/bin/grep -Fq 'xgrammar/libxgrammar_bindings.dylib' "$record" 2>/dev/null; then
          /usr/bin/printf '%s\\n' 'xgrammar/libxgrammar_bindings.dylib,,' >> "$record"
        fi
        "$python" -c 'import xgrammar; print("xgrammar import OK")'
      fi

      # Custom kernels need the final mlx library rpath after Homebrew cleaning.
      kernel_root="$site/omlx/custom_kernels"
      if [ -d "$kernel_root" ]; then
        mlx_lib="$($python -c 'import os, mlx.core; print(os.path.join(os.path.dirname(mlx.core.__file__), "lib"))')"
        for lib in "$kernel_root"/*/_ext*.so "$kernel_root"/*/lib*_kernel_ops.dylib; do
          [ -f "$lib" ] || continue
          if ! /usr/bin/otool -l "$lib" | /usr/bin/grep -Fq "$mlx_lib"; then
            /usr/bin/install_name_tool -add_rpath "$mlx_lib" "$lib"
            /usr/bin/codesign --force --sign - "$lib"
          fi
        done
      fi
    SH
  end

  def verify_custom_kernels(python)
    system python, "-c", <<~PYTHON
      import importlib
      failed = {}
      for package in #{CUSTOM_KERNELS.inspect}:
          fast = importlib.import_module(f"omlx.custom_kernels.{package}.fast")
          if not fast.is_native_available():
              failed[package] = str(fast.import_error())
      assert not failed, failed
    PYTHON
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/omlx --version")
    system libexec/"bin/python", "-c",
           "import spacy; spacy.load('en_core_web_sm')"
    verify_custom_kernels(libexec/"bin/python") if build.with?("custom-kernel")
  end
end
