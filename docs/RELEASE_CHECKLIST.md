# Release Checklist — jaxfne v0.1.0

## No-credential pre-flight (run first, no ~/.pypirc needed)

- [ ] `git checkout dev && git pull --ff-only origin dev`
- [ ] `./scripts/release_rehearsal.sh` — must pass all sections

## TestPyPI gate (requires TestPyPI token in ~/.pypirc)

- [ ] Create `~/.pypirc` with `[testpypi]` section (token from https://test.pypi.org/manage/account/token/)
- [ ] `chmod 600 ~/.pypirc`
- [ ] `./scripts/upload_testpypi.sh`
- [ ] Fresh venv install smoke from TestPyPI:
  ```bash
  pip install --index-url https://test.pypi.org/simple/ \
              --extra-index-url https://pypi.org/simple/ \
              jaxfne==0.1.0
  ```
- [ ] Manual Colab smoke — follow `docs/COLAB_SMOKE_V010.md` Cell 1 + Cell 3

## Real PyPI release (requires TestPyPI gate + Colab smoke to pass)

- [ ] `git checkout main && git pull --ff-only origin main`
- [ ] `git merge --ff-only dev`
- [ ] `git push origin main`
- [ ] `git tag -a v0.1.0 -m "jaxfne v0.1.0 practical OOP core freeze"`
- [ ] `git push origin v0.1.0`
- [ ] `JAXFNE_CONFIRM_REAL_PYPI=1 ./scripts/upload_pypi.sh`
- [ ] Real PyPI venv smoke: `pip install jaxfne==0.1.0`
- [ ] Real Colab smoke — follow `docs/COLAB_SMOKE_V010.md` Cell 2 + Cell 3

## No-token policy

Never commit, print, or paste API tokens. Tokens live only in `~/.pypirc` (mode 600).
