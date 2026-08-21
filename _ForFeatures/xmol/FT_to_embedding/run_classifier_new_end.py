# coding=utf-8
from __future__ import absolute_import, division, print_function

import os
import time
import sys
import tempfile
import numpy as np
import paddle.fluid as fluid
from tqdm import tqdm

xmodel_root = os.environ.get("ADAP_MODEL_ROOT")
if xmodel_root:
    sys.path.append(os.path.join(xmodel_root, "FT_to_embedding"))
else:
    raise RuntimeError("Environment variable ADAP_MODEL_ROOT is not set")

import reader.task_reader_multi as task_reader_multi
from model.ernie import ErnieConfig
from finetune.classifier import create_model_emb, evaluate_emb
from utils.args import print_arguments
from utils.init import init_checkpoint
from finetune_args import parser

args = parser.parse_args()


class MoleculeEmbeddingExtractor:
    """
    Molecule embedding extractor using a single‑process batch mode.
    """
    
    def __init__(self, args):
        self.args = args
        
        # Force batch size ≤ 4 to ensure correctness
        self.batch_size = min(args.batch_size, 4)
        self.max_seq_len = args.max_seq_len
        self.use_multi_gpu = args.is_distributed
        
        if args.batch_size > 4:
            print("Warning: batch_size={} has been automatically reduced to 4 for correctness.".format(args.batch_size))
        
        self._init_model()
    
    def _get_gpu_devices(self):
        if self.args.is_distributed:
            return os.getenv("FLAGS_selected_gpus", "0").split(",")
        return [str(i) for i in range(fluid.core.get_cuda_device_count())]
    
    def _init_model(self):
        """Initialize the ERNIE model and inference program."""
        ernie_config = ErnieConfig(self.args.ernie_config_path)
        
        # Only ask paddle about GPUs when we actually intend to use one:
        # fluid.core.get_cuda_device_count() does not exist in a CPU-only build.
        if self.args.use_cuda:
            gpu_devices = self._get_gpu_devices()
            gpu_id = int(gpu_devices[0]) if gpu_devices else 0
            self.place = fluid.CUDAPlace(gpu_id)
        else:
            self.place = fluid.CPUPlace()
        
        self.reader = task_reader_multi.ClassifyReader(
            vocab_path=self.args.vocab_path,
            label_map_config=self.args.label_map_config,
            max_seq_len=self.args.max_seq_len,
            do_lower_case=self.args.do_lower_case,
            in_tokens=self.args.in_tokens,
            tokenizer=self.args.tokenizer,
            task_type_='emb'
        )
        
        startup_prog = fluid.Program()
        if self.args.random_seed is not None:
            startup_prog.random_seed = self.args.random_seed
        
        self.infer_prog = fluid.Program()
        with fluid.program_guard(self.infer_prog, startup_prog):
            with fluid.unique_name.guard():
                self.infer_pyreader, self.emb_feats, self.feed_targets_name = create_model_emb(
                    self.args, pyreader_name='infer_reader',
                    ernie_config=ernie_config, is_prediction=True
                )
        
        self.infer_prog = self.infer_prog.clone(for_test=True)
        
        self.exe = fluid.Executor(self.place)
        self.exe.run(startup_prog)
        
        if not self.args.init_checkpoint:
            raise ValueError("args 'init_checkpoint' must be set!")
        init_checkpoint(self.exe, self.args.init_checkpoint,
                       main_program=startup_prog, use_fp16=self.args.use_fp16)
    
    def load_molecules(self, data_dir):
        """Load all molecules from the given directory."""
        molecules = []
        if not os.path.exists(data_dir):
            return molecules
        
        drugids = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
        drugids.sort()
        
        for drugid in drugids:
            tsv_file = os.path.join(data_dir, drugid, "{}.tsv".format(drugid))
            if not os.path.exists(tsv_file):
                continue
            try:
                with open(tsv_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        parts = lines[1].strip().split('\t')
                        if len(parts) > 0:
                            molecules.append((drugid, parts[0]))
                    elif len(lines) == 1:
                        molecules.append((drugid, lines[0].strip().split('\t')[0]))
            except Exception as e:
                print("Error reading {}: {}".format(tsv_file, e))
        return molecules
    
    def extract_single(self, smiles):
        """Extract embedding for a single SMILES string."""
        temp_file = None
        try:
            fd, temp_file = tempfile.mkstemp(suffix='.tsv', text=True)
            with os.fdopen(fd, 'w') as f:
                f.write("text_a\tlabel\n")
                f.write("{}\t0\n".format(smiles))
            
            self.infer_pyreader.decorate_tensor_provider(
                self.reader.data_generator(
                    temp_file, batch_size=1, epoch=1, dev_count=1, shuffle=False
                )
            )
            
            savedata = evaluate_emb(
                self.exe, self.infer_prog, self.infer_pyreader,
                {"emb_feats": self.emb_feats, "feed_targets_name": self.feed_targets_name},
                "infer", use_multi_gpu_test=self.use_multi_gpu, flag='final'
            )
            
            drug_representation = np.array(savedata['embed'][0])
            return drug_representation.mean(axis=0).astype(np.float32)
        
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass
    
    def extract_batch(self, smiles_list):
        """Extract embeddings for a batch of SMILES strings."""
        if not smiles_list:
            return []
        
        temp_file = None
        try:
            fd, temp_file = tempfile.mkstemp(suffix='.tsv', text=True)
            with os.fdopen(fd, 'w') as f:
                f.write("text_a\tlabel\n")
                for smiles in smiles_list:
                    f.write("{}\t0\n".format(smiles))
            
            self.infer_pyreader.decorate_tensor_provider(
                self.reader.data_generator(
                    temp_file,
                    batch_size=len(smiles_list),
                    epoch=1,
                    dev_count=1,
                    shuffle=False
                )
            )
            
            savedata = evaluate_emb(
                self.exe, self.infer_prog, self.infer_pyreader,
                {"emb_feats": self.emb_feats, "feed_targets_name": self.feed_targets_name},
                "infer", use_multi_gpu_test=self.use_multi_gpu, flag='final'
            )
            
            embed_data = savedata.get('embed', [])
            batch_size = len(smiles_list)
            embeddings = []
            
            # Handle nested list structure
            for i in range(batch_size):
                if i < len(embed_data):
                    token_list = embed_data[i]
                    if isinstance(token_list, list):
                        token_embeddings = np.array(token_list, dtype=np.float32)
                    else:
                        token_embeddings = token_list
                    
                    if len(token_embeddings.shape) == 2:
                        pooled = np.mean(token_embeddings, axis=0)
                    else:
                        pooled = token_embeddings
                    embeddings.append(pooled.astype(np.float32))
                else:
                    embeddings.append(np.zeros(768, dtype=np.float32))
            
            return embeddings
            
        finally:
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def process_molecules(self, data_dir, output_dir):
        """Process all molecules with progress display."""
        print("=" * 80)
        print("Starting molecule processing")
        print("Data directory: {}".format(data_dir))
        print("Output directory: {}".format(output_dir))
        print("Batch size: {}".format(self.batch_size))
        print("=" * 80)
        
        print("Loading molecule list...")
        molecules = self.load_molecules(data_dir)
        total = len(molecules)
        print("Total molecules found: {:,}".format(total))
        
        if total == 0:
            print("Warning: No molecule files found!")
            return
        
        success_count = 0
        failed_drugs = []
        start_time = time.time()
        
        pbar = tqdm(total=total, desc="Processing molecules", unit="mol")
        
        for i in range(0, total, self.batch_size):
            batch = molecules[i:i+self.batch_size]
            drugids = [item[0] for item in batch]
            smiles_list = [item[1] for item in batch]
            
            embeddings = self.extract_batch(smiles_list)
            
            for drugid, embedding in zip(drugids, embeddings):
                if embedding is not None and not np.isnan(embedding).any():
                    save_dir = os.path.join(output_dir, drugid)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_path = os.path.join(save_dir, "{}.npy".format(drugid))
                    np.save(save_path, embedding)
                    success_count += 1
                else:
                    failed_drugs.append(drugid)
            
            pbar.update(len(batch))
        
        pbar.close()
        
        elapsed = time.time() - start_time
        print("\n" + "=" * 80)
        print("Processing completed!")
        print("Total molecules: {:,}".format(total))
        print("Successfully saved: {:,}".format(success_count))
        print("Failed: {:,}".format(len(failed_drugs)))
        print("Total time: {:.2f} sec ({:.2f} min)".format(elapsed, elapsed/60))
        if elapsed > 0 and success_count > 0:
            print("Average speed: {:.2f} mol/s".format(success_count/elapsed))
        print("=" * 80)
        
        if failed_drugs:
            failed_file = "failed_drugs_{}.txt".format(int(time.time()))
            with open(failed_file, "w") as f:
                for drugid in failed_drugs:
                    f.write("{}\n".format(drugid))
            print("Failed list saved to: {}".format(failed_file))


def main(args):
    """Main entry point."""
    if args.use_cuda:
        gpu_count = fluid.core.get_cuda_device_count()
        print("Detected {} GPU devices".format(gpu_count))
    
    extractor = MoleculeEmbeddingExtractor(args)
    data_dir = args.test_set
    output_dir = data_dir
    
    if not os.path.exists(data_dir):
        print("Error: Directory does not exist: {}".format(data_dir))
        sys.exit(1)
    
    drug_dirs = [d for d in os.listdir(data_dir)
                 if os.path.isdir(os.path.join(data_dir, d))]
    print("Found {} molecule directories".format(len(drug_dirs)))
    
    if len(drug_dirs) == 0:
        print("Warning: No molecule directories found in {}".format(data_dir))
        return
    
    extractor.process_molecules(data_dir, output_dir)


if __name__ == '__main__':
    print_arguments(args)
    main(args)