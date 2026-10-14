"""Guard tests use synthetic archives; they do not run or claim research results."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
import zipfile

REPO=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('entry',REPO/'reproduce.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
def snapshot(root):
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}

class RuntimeGuards(unittest.TestCase):
    def setUp(self):
        self.parent=Path(tempfile.gettempdir()).resolve()
        self.tmp=Path(tempfile.mkdtemp(prefix='caltransfer-guard-',dir=self.parent)).resolve()
        self.repo=self.tmp/'repo';(self.repo/'work/model').mkdir(parents=True)
        (self.repo/'frozen').mkdir()
        (self.repo/'work/model/model.py').write_text('VALUE=1\n')
        (self.repo/'work/model/config.json').write_text('{}')
        (self.repo/'reproduce.py').write_text('# test source\n')
        (self.repo/'requirements.txt').write_text('numpy==2.5.3\n')
        (self.repo/'frozen/aggregate_results.json').write_text('{"tables":{}}')
        self.zip=self.tmp/'data.zip'
        with zipfile.ZipFile(self.zip,'w') as z:z.writestr('data.csv','x\n1\n')
        self.saved=entry.REPO;entry.REPO=self.repo
        self.args=SimpleNamespace(command='prepare',run_dir=self.tmp/'runtime',smoke=False,
                                  oulad_zip=self.zip,xuetangx_zip=None,threads=4,memory='4GB')
    def tearDown(self):
        entry.REPO=self.saved
        # Only this test's newly created, explicitly checked temporary directory is removed.
        if self.tmp.parent!=self.parent or not self.tmp.name.startswith('caltransfer-guard-'):
            raise RuntimeError('Unexpected cleanup path')
        shutil.rmtree(self.tmp)
    def unchanged_after_rejection(self,call):
        before=snapshot(self.args.run_dir)
        with self.assertRaises(ValueError):call()
        self.assertEqual(before,snapshot(self.args.run_dir))
    def test_first_and_continuing_run(self):
        root,cfg,_=entry.init(self.args)
        self.assertEqual(cfg['mode'],'full')
        self.assertEqual(len(cfg['inputs']['oulad_zip']['sha256']),64)
        self.assertEqual(len(cfg['source_manifest_sha256']),64)
        before=snapshot(root);self.args.command='run';self.args.oulad_zip=None
        entry.init(self.args);self.assertEqual(before,snapshot(root))
    def test_same_path_archive_replacement_rejected(self):
        entry.init(self.args)
        with zipfile.ZipFile(self.zip,'w') as z:z.writestr('data.csv','x\n2\n')
        self.unchanged_after_rejection(lambda:entry.init(self.args))
    def test_distributed_source_change_rejected(self):
        entry.init(self.args);(self.repo/'work/model/model.py').write_text('VALUE=2\n')
        self.unchanged_after_rejection(lambda:entry.init(self.args))
    def test_runtime_source_change_rejected(self):
        root,_,_=entry.init(self.args);(root/'work/model/model.py').write_text('VALUE=99\n')
        self.unchanged_after_rejection(lambda:entry.init(self.args))
    def test_quick_cannot_write_over_full(self):
        entry.init(self.args);self.args.command='quick'
        self.unchanged_after_rejection(lambda:entry.quick(self.args))
    def test_full_cannot_reuse_quick(self):
        self.args.command='quick';self.args.oulad_zip=None;entry.init(self.args)
        self.args.command='run';self.unchanged_after_rejection(lambda:entry.init(self.args))
    def test_smoke_cannot_reuse_full(self):
        entry.init(self.args);self.args.smoke=True
        self.unchanged_after_rejection(lambda:entry.init(self.args))
    def test_full_cannot_reuse_smoke(self):
        self.args.smoke=True;entry.init(self.args);self.args.smoke=False
        self.unchanged_after_rejection(lambda:entry.init(self.args))
    def test_quick_rejects_repository_before_write(self):
        self.args.command='quick';self.args.run_dir=self.repo
        self.unchanged_after_rejection(lambda:entry.quick(self.args))
    def test_unverified_directory_rejected(self):
        self.args.run_dir.mkdir();(self.args.run_dir/'user_file.txt').write_text('keep me')
        self.unchanged_after_rejection(lambda:entry.init(self.args))
    def test_legacy_config_rejected(self):
        self.args.run_dir.mkdir();(self.args.run_dir/'run_config.json').write_text('{}')
        self.unchanged_after_rejection(lambda:entry.init(self.args))

class SupportedJobGuards(unittest.TestCase):
    def test_missing_duplicate_failed_and_nonfinite_rejected(self):
        import pandas as pd
        spec=importlib.util.spec_from_file_location('integrity',REPO/'work/xuetangx_experiment4/run_integrity.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        support=pd.DataFrame([{'pair_id':'P000','day':7,'seed':17,'admissible':True}])
        good=pd.DataFrame([{'pair_id':'P000','day':7,'seed':17,'base_model':model,
                           'G_S':.01,'G_T':.02,'G_T_star':.03,'A':.01}
                          for model in ('logistic','lightgbm')])
        self.assertEqual(m.validate_job_coverage(good,support),2)
        for broken in (good.iloc[:1],pd.concat([good,good.iloc[:1]],ignore_index=True),good.assign(A=float('nan'))):
            with self.assertRaises(RuntimeError):m.validate_job_coverage(broken,support)
        with self.assertRaises(RuntimeError):m.validate_job_coverage(good,support,pd.DataFrame([{'error':'deliberate test'}]))

if __name__=='__main__':unittest.main()
