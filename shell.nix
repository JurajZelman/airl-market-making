{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python311
    python311Packages.pip
    python311Packages.uv
    direnv
    git
    git-lfs
    pre-commit
    autoPatchelfHook
    ruff
  ];

  # Fix numpy
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.stdenv.cc.cc
    pkgs.zlib
  ];

  shellHook = ''
    # Set Python version
    export PYTHON_VERSION="3.11"
  '';
}
