# Publishing checklist (maintainer-only)

Everything below is prepared and committed **locally**; nothing has been
pushed to GitHub. This is the exact sequence to get it live at
https://github.com/idrblab/AdaptiveHIT as `yimiaozhu`.

**Architecture note:** `base_model/*` submodules are pinned directly at the
original authors' own upstream commits (`samsledje/ConPLex_dev`,
`peizhenbai/DrugBAN`, `lifanchen-simm/TransformerCPI`,
`GIST-CSBL/DeepConv-DTI`) -- not forks. AdaptiveHIT's own additions to each
live only as a patch under `patches/<name>.patch`, applied locally by
`run.sh`. That means there's nothing to fork or push for the base models --
you only ever publish this one repo.

## 0. Install prerequisites (done -- no sudo needed)

`gh` and `git-lfs` are now installed under `~/.local/bin` (no root
required -- official prebuilt binaries downloaded straight from each
project's GitHub Releases and extracted there, not via `apt`):

```bash
gh --version        # gh version 2.97.0
git lfs version      # git-lfs/3.7.1
```

`~/.local/bin` was added to `PATH` via `~/.bashrc`; open a new terminal (or
`source ~/.bashrc`) for it to take effect. `git lfs install` has already
been run once (registers the LFS filter in `~/.gitconfig`, also no root
needed) -- confirmed working: `git status` on the checkpoint/`xmol_weights`
files is now clean (they were never actually modified; git just couldn't
compare LFS pointers to their real binaries before `git-lfs` was
installed), and `git lfs ls-files` correctly lists all tracked files.

Only `gh auth login` is still up to you (interactive):
```bash
gh auth login   # authenticate as yimiaozhu; choose HTTPS or SSH per your preference
```

## 1. Push the main AdaptiveHIT repo

```bash
cd /mnt/hdd/data/zym/AOEDrug/AdaptiveHIT
gh repo create idrblab/AdaptiveHIT --public --source=. --remote=origin
git push -u origin main
git push origin --tags   # if any tags were created
```

`git lfs` tracking is already configured in this repo (`checkpoints/**/*.model`,
`checkpoints/**/*.pth`, `data_adapter/xmol_weights/**` via `.gitattributes`)
-- since `git-lfs` is installed (step 0), `git push` will automatically also
push the LFS objects. Confirm afterward:

```bash
git lfs ls-files          # should list the checkpoint + X-Mol weight files
```

GitHub's free LFS quota is 1GB storage / 1GB bandwidth per month per
repository by default -- this repo's LFS objects total ~423MB
(checkpoints ~96MB + X-Mol weights ~327MB), so a single push fits, but
enable GitHub's paid LFS data pack if the repo gets cloned frequently.

## 2. Verify the public clone works

This is the real test: a fresh clone with `--recurse-submodules` pulls each
`base_model/*` submodule straight from its real upstream repo (no fork
needed), then `run.sh` applies `patches/*.patch` on top automatically.

```bash
cd /tmp
git clone --recurse-submodules https://github.com/idrblab/AdaptiveHIT.git
cd AdaptiveHIT
bash run.sh
conda activate adaptivehit
python meta_learner/predict.py \
    --input_dir data/toy_dataset/end_merged \
    --output_dir /tmp/out \
    --weights_dir data/toy_dataset/weights \
    --strategies average vote-all-1 weighted_logistic_balanced \
    --eval
```

## 3. Optional: clean up the local staging directory

`/mnt/hdd/data/zym/AOEDrug/_release_staging/` (the old forked-repo staging
copies from before this switched to the pristine-submodule-plus-patch
approach) is no longer needed by anything in this repo -- safe to delete if
you want to reclaim the disk space.
