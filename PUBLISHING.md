# Publishing checklist (maintainer-only)

Everything below is prepared and committed **locally**; nothing has been
pushed to GitHub. This is the exact sequence to get it live under the
`idrb` organization as `yimiaozhu`. `gh` is already installed locally
(`~/.local/bin/gh`) but not yet authenticated -- do that first.

```bash
gh auth login   # authenticate as yimiaozhu; choose HTTPS or SSH per your preference
```

## 1. Push the 4 forked base-model repos

Each is a fully-prepared local git repo (clean upstream history + a commit
adding AdaptiveHIT's integration scripts/patches) sitting in
`/home/zhuyimiao/AOEDrug/_release_staging/<name>/`. Create each as a repo
under `idrb` and push:

```bash
cd /home/zhuyimiao/AOEDrug/_release_staging

for name in ConPLex_dev DrugBAN TransformerCPI DeepConv-DTI; do
    cd "$name"
    gh repo create "idrb/$name" --public --source=. --remote=origin
    git push -u origin "$(git branch --show-current)"   # TransformerCPI/DeepConv-DTI default to "master", the other two to "main"
    cd ..
done
```

If `idrb/<name>` already exists (e.g. a prior fork), use
`git remote add origin git@github.com:idrb/<name>.git` and `git push -u
origin main` instead of `gh repo create`.

## 2. Wire the main repo's submodules to the real URLs

Right now `.gitmodules` in `/home/zhuyimiao/AOEDrug/AdaptiveHIT` points at
local `file://` paths (used only so this repo could be verified end-to-end
before anything was pushed). Repoint each to the real GitHub URL you just
created:

```bash
cd /home/zhuyimiao/AOEDrug/AdaptiveHIT

git submodule set-url base_model/ConPLex_dev      https://github.com/idrb/ConPLex_dev.git
git submodule set-url base_model/DrugBAN          https://github.com/idrb/DrugBAN.git
git submodule set-url base_model/TransformerCPI   https://github.com/idrb/TransformerCPI.git
git submodule set-url base_model/DeepConv-DTI     https://github.com/idrb/DeepConv-DTI.git

git submodule sync
git add .gitmodules
git commit -m "Point submodules at published idrb/* forks"
```

## 3. Push the main AdaptiveHIT repo

```bash
cd /home/zhuyimiao/AOEDrug/AdaptiveHIT
gh repo create idrb/AdaptiveHIT --public --source=. --remote=origin
git push -u origin main
git push origin --tags   # if any tags were created
```

`git lfs` is already initialized in this repo (`checkpoints/**/*.model`,
`checkpoints/**/*.pth`, `data_adapter/xmol_weights/**` are tracked via
`.gitattributes`) -- `git push` will automatically also push the LFS
objects. Confirm afterward:

```bash
git lfs ls-files          # should list the checkpoint + X-Mol weight files
```

GitHub's free LFS quota is 1GB storage / 1GB bandwidth per month per
repository by default -- this repo's LFS objects total ~423MB
(checkpoints ~96MB + X-Mol weights ~327MB), so a single push fits, but
enable GitHub's paid LFS data pack if the repo gets cloned frequently.

## 4. Verify the public clone works

```bash
cd /tmp
git clone --recurse-submodules https://github.com/idrb/AdaptiveHIT.git
cd AdaptiveHIT
conda env create -f environment.yml
conda activate adaptivehit
python meta_learner/predict.py --input_dir data/toy_dataset --output_dir /tmp/out --eval
```

## 5. Optional: clean up the local staging directory

Once pushed and verified, `/home/zhuyimiao/AOEDrug/_release_staging/` is no
longer needed (the same content now lives in the pushed `idrb/*` repos) --
safe to delete if you want to reclaim the disk space.
