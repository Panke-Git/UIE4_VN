"""Diagnostic software tests. Synthetic fixtures never establish real DUO findings."""
from copy import deepcopy
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from src.downstream_duo.diagnostics import common
from src.downstream_duo.diagnostics.common import (expected_coco, labels_xyxy, same_boxes, rgb, rgb_tensor,
    save_tensor, pixel_difference, snapshot, check_snapshot, image_statistics)
from src.downstream_duo.diagnostics.compare_full_vs_tiled import (fixed_sample,freeze_sample,boundary_statistics,
    infer_one,conditioning_trace)
from src.downstream_duo.diagnostics.identity_control import identity_gate
from src.downstream_duo.diagnostics.report import decision,write_report
from src.downstream_duo.model_inference_v16 import enhance_tensor,starts
from src.downstream_duo.smoke import tiny_v16_config
from src.downstream_duo.utils import read_json,write_json,sha256


def test_rgb_bgr_and_identity_roundtrip(tmp_path):
    a = np.array([[[255,0,17],[11,133,241]],[[1,2,3],[255,255,255]]],dtype=np.uint8)
    source,output = tmp_path/'source.png',tmp_path/'identity.png'
    Image.fromarray(a).save(source)
    save_tensor(rgb_tensor(rgb(source)),output)
    assert np.array_equal(rgb(output),a)
    assert not np.array_equal(rgb(output),a[...,::-1])
    assert pixel_difference(rgb(output),a)['identical']
    assert pixel_difference(rgb(output),a[...,::-1])['mae_uint8'] > 1


def test_uint8_encoding_range_rounding_nonfinite(tmp_path):
    tensor = torch.tensor([0.,.5,1.]).view(1,3,1,1)
    out = save_tensor(tensor,tmp_path/'pixel.png')
    assert out.tolist() == [[[0,128,255]]]
    assert out.shape == (1,1,3)
    for value in (float('nan'),float('inf'),-1.,2.):
        with pytest.raises(ValueError):
            save_tensor(torch.full((1,3,2,4),value),tmp_path/'bad.png')


def coco_fixture():
    doc = {'images':[{'id':7,'file_name':'a.jpg','width':100,'height':50},
                     {'id':91,'file_name':'b.jpg','width':80,'height':40}],
           'categories':[{'id':33,'name':'class_b'},{'id':9,'name':'class_a'}],
           'annotations':[{'id':1,'image_id':7,'category_id':33,'bbox':[10,5,20,15]}]}
    meta = {'mapping':{'names':['class_a','class_b'],'coco_to_yolo':{'9':0,'33':1}},'bbox_policy':'strict'}
    return doc,meta


def test_coco_id_class_gt_coordinate_alignment(tmp_path):
    doc,meta = coco_fixture()
    images,boxes,changes = expected_coco(doc,meta)
    assert set(images) == {7,91} and changes == [] and boxes[91] == []
    path = tmp_path/'label.txt'
    path.write_text('1 0.2 0.25 0.2 0.3\n')
    assert same_boxes(labels_xyxy(path,100,50),boxes[7])
    assert not same_boxes(labels_xyxy(path,100,50),[(1,11,5,31,20)])
    assert not same_boxes(labels_xyxy(path,100,50),[(0,10,5,30,20)])
    assert not same_boxes(labels_xyxy(path,80,40),boxes[7])


def test_bbox_roundtrip_tied_x_does_not_depend_on_sort_order():
    expected = [(0,10.,5.,20.,15.), (0,10.,30.,40.,50.)]
    actual = [(0,10.+1e-9,5.,20.,15.), (0,10.-1e-9,30.,40.,50.)]
    assert not np.allclose(sorted(actual),sorted(expected),rtol=0,atol=1e-5)
    assert same_boxes(sorted(actual),sorted(expected))
    assert same_boxes(list(reversed(actual)),expected)
    assert not same_boxes([(0,10.,5.,21.,15.),actual[1]],expected)
    assert not same_boxes([(1,*actual[0][1:]),actual[1]],expected)


def test_bbox_matching_requires_one_to_one_not_reused_nearest_box():
    expected = [(0,10.,5.,20.,15.), (0,40.,30.,50.,45.)]
    assert not same_boxes([expected[0],expected[0]],expected)
    assert not same_boxes(expected[:1],expected)
    assert same_boxes([],[])


def test_bbox_matching_uses_augmenting_path_for_ambiguous_candidates():
    tol = 1e-5
    def shifted(amount):
        return (0,10.+amount,5.+amount,20.+amount,15.+amount)
    expected = [shifted(0),shifted(1.5*tol)]
    actual = [shifted(.75*tol),shifted(-.5*tol)]
    # The first actual box matches either reference; the second only the first.
    assert same_boxes(actual,expected,tol)
    assert same_boxes(list(reversed(actual)),list(reversed(expected)),tol)
    assert not same_boxes(actual,expected,tol/10)


def test_bbox_matching_rejects_nonfinite_or_invalid_shape():
    valid = [(0,1.,2.,3.,4.)]
    assert not same_boxes([(0,float('nan'),2.,3.,4.)],valid)
    assert not same_boxes([(0,1.,2.,3.)],valid)
    with pytest.raises(ValueError):same_boxes(valid,valid,-1)


@pytest.mark.parametrize('field,value',[('image_id',91),('source_file','b.jpg'),('file','7/other.png'),
    ('file','../outside.png'),('file','/absolute.png'),('file','C:/outside.png'),('width',99),('height',49),('official_split','train')])
def test_strict_file_id_dimensions_and_no_path_escape(field,value):
    official={'id':7,'file_name':'a.jpg','width':100,'height':50}
    row={'image_id':7,'source_file':'a.jpg','file':'7/a.png','width':100,'height':50,'official_split':'test'}
    assert common.validate_image_binding(row,official)==Path('7/a.png')
    row[field]=value
    with pytest.raises(ValueError):common.validate_image_binding(row,official)


def test_distinct_annotation_ids_with_duplicate_boxes_rejected():
    doc,meta=coco_fixture()
    doc['annotations'].append({**doc['annotations'][0],'id':2})
    with pytest.raises(ValueError,match='Duplicate'):expected_coco(doc,meta)


def test_new_diagnostics_contain_no_training_calls():
    import ast
    for path in Path(common.__file__).parent.glob('*.py'):
        if path.name == 'p1.py':
            continue  # Explicitly gated optional matched-domain training, outside P0.
        tree=ast.parse(path.read_text())
        forbidden=[node.func.attr for node in ast.walk(tree) if isinstance(node,ast.Call)
                   and isinstance(node.func,ast.Attribute) and node.func.attr in ('train','backward','step')]
        assert not forbidden, (path,forbidden)
    runner=(Path(common.__file__).parent/'run_p0.py').read_text()
    assert 'from .p1' not in runner and 'diagnostics.p1' not in runner


@pytest.mark.parametrize('kind',['duplicate_image','duplicate_annotation','unknown_image','unknown_class','wrong_mapping','nan','nonpositive'])
def test_invalid_official_records_rejected(kind):
    doc,meta = coco_fixture()
    if kind == 'duplicate_image':doc['images'].append(dict(doc['images'][0]))
    if kind == 'duplicate_annotation':doc['annotations'].append(dict(doc['annotations'][0]))
    if kind == 'unknown_image':doc['annotations'][0]['image_id'] = 123
    if kind == 'unknown_class':doc['annotations'][0]['category_id'] = 55
    if kind == 'wrong_mapping':meta['mapping']['coco_to_yolo']['9'] = 1
    if kind == 'nan':doc['annotations'][0]['bbox'][0] = float('nan')
    if kind == 'nonpositive':doc['annotations'][0]['bbox'][2] = 0
    with pytest.raises(ValueError):expected_coco(doc,meta)


def test_recorded_clipping_only_and_empty_label(tmp_path):
    doc,meta = coco_fixture()
    doc['annotations'][0]['bbox'] = [90,5,11,15]
    with pytest.raises(ValueError):expected_coco(doc,meta)
    meta['bbox_policy']='clip'
    _,boxes,changes = expected_coco(doc,meta)
    assert boxes[7] == [(1,90,5,100,20)] and changes == [1]
    doc['annotations'][0]['bbox'] = [10,50,20,1]
    with pytest.raises(ValueError):expected_coco(doc,meta)
    meta['empty_after_clip']='drop'
    assert expected_coco(doc,meta)[1][7] == []
    path=tmp_path/'empty.txt';path.write_text('')
    assert labels_xyxy(path,100,50) == []


@pytest.mark.parametrize('line',['-1 .5 .5 .2 .2','0 .5 .5 0 .2','0 nan .5 .2 .2','0 .9 .5 .4 .2','0 .5 .5 .2 .2\n0 .5 .5 .2 .2'])
def test_bad_yolo_labels_rejected(tmp_path,line):
    path=tmp_path/'bad.txt';path.write_text(line)
    with pytest.raises(ValueError):labels_xyxy(path,100,50)


def test_fixed_sample_is_reproducible_covers_classes_and_frozen(tmp_path):
    rows=[{'image_id':i,'width':80+(i%3)*16,'height':64+(i%2)*16} for i in range(100)]
    boxes={i:[(i%4,0,0,10,10)] for i in range(100)}
    ids=fixed_sample(rows,boxes,32)
    assert len(ids)==len(set(ids))==32
    assert ids == fixed_sample(list(reversed(rows)),dict(reversed(list(boxes.items()))),32)
    assert {boxes[i][0][0] for i in ids} == {0,1,2,3}
    path=tmp_path/'sample_ids.json';freeze_sample(path,{'seed':3520,'image_ids':ids})
    before=sha256(path)
    with pytest.raises(FileExistsError):freeze_sample(path,{'image_ids':[99]})
    assert sha256(path)==before
    with pytest.raises(ValueError):fixed_sample(rows,boxes,32,seed=1)


def test_tile_placement_and_weight_normalization_independent_numpy():
    torch.set_num_threads(2)
    class LocalCoordinate(torch.nn.Module):
        def forward(self,x):
            values=torch.linspace(0,1,x.shape[-1],device=x.device)
            return values.view(1,1,1,-1).expand_as(x)
    model=LocalCoordinate().eval()
    x=torch.zeros(1,3,35,51)
    actual=enhance_tensor(model,x,tile_size=32,overlap=8).numpy()
    total=np.zeros((35,51),dtype=np.float64);denom=np.zeros_like(total)
    wy=np.maximum(np.hanning(32),.01);wx=wy.copy();weight=wy[:,None]*wx[None,:]
    # Independently enumerated tile origins rather than reusing the production starts().
    for y in (0,3):
        for left in (0,19):
            total[y:y+32,left:left+32] += np.linspace(0,1,32)[None,:]*weight
            denom[y:y+32,left:left+32] += weight
    assert denom.min() > 0
    assert np.allclose(actual[0,0],total/denom,atol=3e-7)
    assert not np.allclose(actual,enhance_tensor(model,x).numpy())
    ramp=torch.rand(1,3,35,51)
    assert torch.allclose(enhance_tensor(torch.nn.Identity().eval(),ramp,tile_size=32,overlap=8),ramp,atol=3e-7)


def test_boundary_control_does_not_fabricate_seams():
    ramp=np.broadcast_to(np.arange(70,dtype=np.uint8)[None,:,None],(45,70,3)).copy()
    stats=boundary_statistics(ramp,ramp,32,8)
    assert all(v['boundary_gradient_change_uint8']==0 for v in stats['axes'])
    assert all(v['nonboundary_gradient_change_uint8']==0 for v in stats['axes'])
    bad=ramp.copy();bad[:,24:,:]=np.minimum(bad[:,24:,:].astype(np.uint16)+30,255).astype(np.uint8)
    observed=boundary_statistics(ramp,bad,32,8)
    assert observed['axes'][1]['boundary_gradient_change_uint8'] > 0
    assert 'do not establish' in observed['caveat']


def test_oom_record_is_explicit_no_fake_native_output(tmp_path):
    from src.v16.models.network import build_model
    model=build_model(tiny_v16_config()['model']).eval().requires_grad_(False)
    def oom(x):raise torch.OutOfMemoryError('SIMULATED software OOM, not GPU evidence')
    model.forward=oom
    image,record=infer_one(model,torch.zeros(1,3,32,48),tmp_path/'native.png','cpu',0,0)
    assert image is None and record['status']=='oom'
    assert 'SIMULATED' in record['error'] and record['input_hw']==[32,48]
    assert not (tmp_path/'native.png').exists()


def test_conditioning_hooks_capture_per_call_coordinates_and_anchor():
    from src.v16.models.network import build_model
    model=build_model(tiny_v16_config()['model']).eval().requires_grad_(False)
    with conditioning_trace(model) as trace:
        enhance_tensor(model,torch.rand(1,3,35,51),tile_size=32,overlap=8)
    assert len(trace)==4
    assert all(row['coordinate_hw']==[32,32] for row in trace)
    assert all(len(row['anchor_b'][0])==3 and len(row['global_gap_g'][0])==2 for row in trace)
    assert not model.uicf._forward_pre_hooks and not model.uicf.chromatic_anchor._forward_hooks


def test_integrity_guard_detects_mutations_and_added_caches(tmp_path):
    p=tmp_path/'protected';p.mkdir();f=p/'label.txt';f.write_text('original')
    before=snapshot([p]);assert check_snapshot(before)['status']=='pass'
    (p/'new.cache').write_text('unexpected')
    assert check_snapshot(before)['new_files_in_protected_inputs']
    f.write_text('changed')
    assert str(f.resolve()) in check_snapshot(before)['changed_or_removed']


def test_output_guard_and_no_overwrite(tmp_path,monkeypatch):
    monkeypatch.setattr(common,'DIAGNOSTICS',tmp_path/'diagnostics')
    with pytest.raises(ValueError):common.fresh_output(tmp_path/'elsewhere')
    created=common.fresh_output(tmp_path/'diagnostics/new')
    with pytest.raises(FileExistsError):common.fresh_output(created)
    with pytest.raises(ValueError):common.fresh_output(tmp_path/'diagnostics/input/run',[tmp_path/'diagnostics/input'])


def test_identity_alarm_stops_other_attribution_and_keeps_negative_main_result(tmp_path):
    baseline=dict(precision=.8,recall=.7,map50=.867808,map50_95=.698035)
    assert identity_gate(baseline,baseline,True)['status']=='pass'
    bad={**baseline,'map50':baseline['map50']-.01}
    gate=identity_gate(bad,baseline,True)
    assert gate['status']=='stop'
    assert identity_gate(baseline,baseline,False)['status']=='stop'
    assert decision({'status':'pass'},{'status':'stop'}, {}, {})['code']=='stop_identity_pipeline'
    assert decision({'status':'pass'},{'status':'pass'}, {'complete':True,'native_success_ids':[]}, {})['code']=='native_unavailable_cause_undetermined'
    write_json(tmp_path/'audit/audit_protocol.json',{'status':'unknown'})
    summary=write_report(tmp_path)
    assert summary['given_negative_results']['full']['map50']==pytest.approx(.5821621094901419)
    assert summary['p1_executed'] is False


def test_native_single_tile_cannot_rule_out_large_image_tiling():
    comparison={'complete':True,'native_success_ids':[1],'sample_ids':[1,2],
        'native_success_multitile_ids':[],
        'records':[{'image_id':1,'existing_tiled_vs_native':{'mae_uint8':0},'existing_vs_replayed_tiled':{'max_error_uint8':0}},
                   {'image_id':2,'existing_vs_replayed_tiled':{'max_error_uint8':0}}]}
    raw=dict(precision=.8,recall=.7,map50=.8,map50_95=.6)
    native={**raw,'map50':.5,'map50_95':.4}
    subset={'sets':{'native_common':{'metrics':{'raw':raw,'tiled':native,'native':native}}}}
    result=decision({'status':'pass'},{'status':'pass'},comparison,subset)
    assert result['code']=='native_evidence_only_single_tile_images'


def test_pixel_statistics_and_psnr_meaning():
    a=np.zeros((3,4,3),dtype=np.uint8);b=np.full_like(a,255)
    assert image_statistics(a)['black_pixel_fraction']==1
    assert image_statistics(b)['white_pixel_fraction']==1
    assert pixel_difference(a,b)['psnr_between_predictions_db']==0
    assert pixel_difference(a,a)['psnr_between_predictions_db'] is None


@pytest.mark.skipif(not os.getenv('DUO_DIAGNOSTICS_SMOKE_ROOT'),reason='Optional existing no-training synthetic integration receipt')
def test_actual_no_training_software_smoke_receipt():
    root=Path(os.environ['DUO_DIAGNOSTICS_SMOKE_ROOT'])
    audit=read_json(root/'audit/audit_protocol.json')
    identity=read_json(root/'identity/identity_control.json')
    compare=read_json(root/'full_vs_tiled/full_vs_tiled.json')
    summary=read_json(root/'diagnosis_summary.json')
    assert audit['synthetic'] is True and audit['status']=='pass'
    assert identity['pixel_exact'] and identity['status']=='pass'
    assert compare['complete'] and compare['native_success_ids']==compare['sample_ids']
    assert set(read_json(root/'full_vs_tiled/subset_metrics.json')['sets']['native_common']['metrics'])=={'raw','tiled','native'}
    assert summary['input_integrity']['status']=='pass' and summary['p1_executed'] is False


def test_p1_split_membership_allows_numeric_test_id_reuse():
    from src.downstream_duo.diagnostics.p1 import validate_partition
    meta={'seed':3520,'splits':{
        'train':[{'image_id':1,'file':'1/a.png','official_split':'train'}],
        'val':[{'image_id':2,'file':'2/b.png','official_split':'train'}],
        'test':[{'image_id':1,'file':'1/c.png','official_split':'test'}]}}
    validate_partition(meta)
    wrong=deepcopy(meta);wrong['splits']['val'][0]['official_split']='test'
    with pytest.raises(ValueError):validate_partition(wrong)
    wrong=deepcopy(meta);wrong['splits']['val'][0]['image_id']=1
    with pytest.raises(ValueError):validate_partition(wrong)


@pytest.mark.parametrize('key,value',[('amp',True),('resume',True),('cache',True),('epochs',81),('seed',1),('imgsz',256),('data','test.yaml')])
def test_p1_training_configuration_is_locked(key,value):
    from src.downstream_duo.diagnostics.p1 import validate_training_config
    config={'amp':False,'resume':False,'cache':False,'epochs':80,'seed':3520,'imgsz':640}
    validate_training_config(config)
    with pytest.raises(ValueError):validate_training_config({**config,key:value})


def test_p1_initial_weights_must_match_original_hash(tmp_path):
    from src.downstream_duo.diagnostics.p1 import validate_initial_weights
    file=tmp_path/'initial.pt';file.write_bytes(b'software fixture only')
    validate_initial_weights(file,sha256(file))
    with pytest.raises(ValueError):validate_initial_weights(file,'0'*64)
    with pytest.raises(ValueError):validate_initial_weights(file,None)
    yaml=tmp_path/'initial.yaml';yaml.write_text('test: only')
    with pytest.raises(ValueError):validate_initial_weights(yaml,sha256(yaml))


def test_p1_training_needs_explicit_opt_in_and_one_attempt(tmp_path):
    from src.downstream_duo.diagnostics.p1 import authorize_training
    with pytest.raises(ValueError):authorize_training(tmp_path,False)
    assert not (tmp_path/'detector_training_attempt.lock').exists()
    authorize_training(tmp_path,True)
    with pytest.raises(FileExistsError):authorize_training(tmp_path,True)


def test_p1_locks_helper_source_files_too():
    from src.downstream_duo.diagnostics.p1 import source_hashes
    sources=source_hashes()
    assert str(Path(common.__file__).resolve()) in sources
    assert str((Path(common.__file__).parent/'evaluator.py').resolve()) in sources


def test_p1_enhanced_data_geometry_and_label_hash_guard(tmp_path):
    from src.downstream_duo.diagnostics.p1 import verify_enhanced_train
    meta={'splits':{}}
    records=[]
    for split,iid in [('train',1),('val',2)]:
        image=tmp_path/f'enhanced/images/{split}_enhanced/{iid}/a.png'
        label=tmp_path/f'enhanced/labels/{split}_enhanced/{iid}/a.txt'
        image.parent.mkdir(parents=True);label.parent.mkdir(parents=True)
        Image.new('RGB',(7,5),(10,20,30)).save(image);label.write_text('')
        row={'image_id':iid,'file':f'{iid}/a.png','width':7,'height':5,'label_sha256':sha256(label)}
        meta['splits'][split]=[row]
        records.append({'split':split,'image_id':iid,'file':row['file'],'status':'ok','sha256':sha256(image)})
    manifest={'complete':True,'records':records,'checkpoint_sha256':'fixed','tile_size':32,'overlap':8}
    write_json(tmp_path/'enhanced/enhancement_manifest.json',manifest)
    config={'uie_checkpoint_sha256':'fixed','tile_size':32,'overlap':8}
    verify_enhanced_train(tmp_path,config,meta)
    label.write_text('0 .5 .5 .2 .2\n')
    with pytest.raises(ValueError):verify_enhanced_train(tmp_path,config,meta)


def test_p1_shell_does_not_start_without_confirmation():
    import subprocess
    script=Path(common.PROJECT)/'scripts/downstream_duo/diagnostics/run_p1.sh'
    result=subprocess.run(['bash',str(script),'not-a-run','--uie-python','missing-python'],capture_output=True,text=True)
    assert result.returncode==2 and 'nothing started' in result.stderr


@pytest.mark.skipif(not os.getenv('DUO_P1_SMOKE_ROOT'),reason='Optional completed P1 synthetic integration receipt')
def test_p1_actual_synthetic_train_and_two_evaluations_receipt():
    base=Path(os.environ['DUO_P1_SMOKE_ROOT'])
    root=base/'synthetic_p1'
    p=read_json(root/'plan.json');summary=read_json(root/'p1_summary.json')
    registry=read_json(root/'detector/registry.json')
    assert p['synthetic'] is True and summary['synthetic'] is True and summary['status']=='complete'
    assert registry['status']=='complete' and registry['train']['epochs']==1
    assert registry['initial_weights_sha256']==p['initial_weights_sha256']
    a=read_json(root/'evaluation/raw/metrics_raw.json');b=read_json(root/'evaluation/full/metrics_full.json')
    assert a['weights_sha256']==b['weights_sha256']==sha256(registry['weights'])
    assert a['identity']==b['identity']==p['identity'] and a['test_count']==b['test_count']==3
    assert summary['input_integrity']['status']=='pass'
    assert read_json(base/'original_inputs_after.json')['status']=='pass'
