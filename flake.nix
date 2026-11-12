{
  description = "semantic-spotter: detector de código duplicado (T1–T4) sobre repos git";

  # Los nombres de los inputs son los mismos que usa inity, para que el
  # consumidor pueda hacer `follows` de los cuatro y no duplicarlos en su lock.
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
      ...
    }:
    let
      inherit (nixpkgs) lib;

      systems = [
        "x86_64-linux" # 64-bit Intel/AMD Linux
        "aarch64-linux" # 64-bit ARM Linux (CodeBuild Graviton)
        "aarch64-darwin" # 64-bit ARM macOS
      ];
      forAllSystems = lib.genAttrs systems;

      # uv2nix lee uv.lock y pyproject.toml al evaluar. El árbol que ve Nix es el
      # del repo git: los archivos ignorados (como .env) no entran al store.
      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

      # Ruedas precompiladas siempre que existan: no se compila tree-sitter,
      # lancedb ni pyarrow desde fuente. En linux, uv2nix ya agrega
      # autoPatchelfHook y las librerías de manylinux a cada rueda.
      overlay = workspace.mkPyprojectOverlay {
        sourcePreference = "wheel";
      };

      editableOverlay = workspace.mkEditablePyprojectOverlay {
        root = "$REPO_ROOT";
      };

      # Solo lo que hace falta para construir el paquete (uv_build exige el
      # README que declara pyproject.toml).
      packageSrc = lib.fileset.toSource {
        root = ./.;
        fileset = lib.fileset.unions [
          ./pyproject.toml
          ./README.md
          ./src
        ];
      };

      # Lo que necesita pytest: la configuración de pyproject.toml y los tests.
      testSrc = lib.fileset.toSource {
        root = ./.;
        fileset = lib.fileset.unions [
          ./pyproject.toml
          ./tests
        ];
      };

      pythonSets = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          python = pkgs.python314;

          pyprojectOverrides = final: prev: {
            semantic-spotter = prev.semantic-spotter.overrideAttrs (old: {
              src = packageSrc;

              passthru = old.passthru // {
                tests =
                  let
                    # Entorno con el grupo de dependencias dev (pytest).
                    virtualenv = final.mkVirtualEnv "semantic-spotter-test-env" workspace.deps.all;
                  in
                  (old.passthru.tests or { })
                  // {
                    pytest = pkgs.stdenv.mkDerivation {
                      name = "${final.semantic-spotter.name}-pytest";
                      src = testSrc;
                      nativeBuildInputs = [
                        virtualenv
                        pkgs.git
                      ];
                      dontConfigure = true;
                      buildPhase = ''
                        runHook preBuild
                        # Los tests crean repos git y hacen commits: git necesita
                        # un HOME escribible. La identidad la fija cada repo.
                        export HOME="$TMPDIR"
                        export GIT_CONFIG_NOSYSTEM=1
                        pytest -p no:cacheprovider tests
                        runHook postBuild
                      '';
                      installPhase = ''
                        runHook preInstall
                        touch $out
                        runHook postInstall
                      '';
                    };
                  };
              };
            });
          };
        in
        (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope
          (
            lib.composeManyExtensions [
              pyproject-build-systems.overlays.wheel
              overlay
              pyprojectOverrides
            ]
          )
      );
    in
    {
      packages = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          pythonSet = pythonSets.${system};
          inherit (pkgs.callPackages pyproject-nix.build.util { }) mkApplication;
        in
        {
          # Solo el binario `semantic-spotter`: mkApplication deja fuera el
          # intérprete, los scripts de activación y pyvenv.cfg del entorno.
          default = mkApplication {
            venv = pythonSet.mkVirtualEnv "semantic-spotter-env" workspace.deps.default;
            package = pythonSet.semantic-spotter;
          };
        }
      );

      apps = forAllSystems (system: {
        default = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/semantic-spotter";
          meta.description = "Detector de código duplicado sobre repos git";
        };
      });

      checks = forAllSystems (
        system:
        let
          pythonSet = pythonSets.${system};
        in
        {
          default = pythonSet.semantic-spotter.passthru.tests.pytest;
        }
      );

      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          pythonSet = pythonSets.${system}.overrideScope editableOverlay;
          virtualenv = pythonSet.mkVirtualEnv "semantic-spotter-dev-env" workspace.deps.all;
        in
        {
          default = pkgs.mkShell {
            packages = [
              virtualenv
              pkgs.uv
              pkgs.git
            ];
            env = {
              UV_NO_SYNC = "1";
              UV_PYTHON = pythonSet.python.interpreter;
              UV_PYTHON_DOWNLOADS = "never";
            };
            shellHook = ''
              unset PYTHONPATH
              export REPO_ROOT=$(git rev-parse --show-toplevel)
            '';
          };
        }
      );
    };
}
