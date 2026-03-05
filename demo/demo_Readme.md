Small vesion, just for 1000 ramdon choose
文件夹1. Generation
这个从原始SIE lens + shear mockdata里生成多极钜参数, 运行时间使用4张A100并行24进程 24h; 不用的话会很久

整个simulation light cone 产生 4张A100 需要2ds
跑mcmc得到足够Rcusp-phi需要 2--3weeks

demo仅读取这个文件夹产生的结果, 结果可以直接zenodo下载

```sh
python demo/generate_multipule_mock_catalog.py

python lib/compute_lensing_for_mock_catalog.py \
  --sim-idx 0 \
  --num-sim 1000 \
  --fix 1 \
  --nnn 1000 \
  --fits demo/Data/lensed_qso_mock_multipole_temp.fits \


python lib/merge_calc_results_to_fits.py \
  --fits demo/Data/lensed_qso_mock_multipole_temp.fits \
  --json-dir demo/Data/Data_json \
  --out-fits demo/Data/lensed_qso_mock_multipole.fits


python lib/select_cusp_lens_systems.py \
  --fits demo/Data/lensed_qso_mock_multipole.fits \
  --target-image-number 5 \
  --max-images 5 \
  --Rcusp 1 \
  --phi 140 \
  --sym 5 \
  --out-fits demo/Data/cusp_all_observable_multipole.fits
```
ps: 1000个随机系统数量太少, 一个Cusp都不会有, 所以直接使用full simulation 跑好的程序, 具体分析见inspect_catalog_structure


## Quick demo (2 random systems)
```sh
python demo/run_two_mock_systems.py \
  --fits Data/cusp_all_observable_multipole.fits \
  --num-systems 2 \
  --mode both
```

Note: Apple Metal is experimental and may fail for float64. Use CPU by default:
```sh
python demo/run_two_mock_systems.py --mode both --compute cpu
```
光锥计算速度受生成的子晕影响 
apple M2 一个系统 FDM需要大约10min , SIDM, CDM 晕各需要大约15min
