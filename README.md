# Train

```bash
python3 train.py --config-path=src/configs/local --config-name=big.yaml
```

# Test

Download checkpoint from the

https://drive.google.com/file/d/1ojAFd3P03ua7UCskLnhAxSyZ3pyGl8E5/view?usp=sharing

and move it to the root of this project.

В `src/configs/test.json` в графе data надо указать путь до тестового датасета,
в котором будут лежать файлы реф, микс, таргет в формате типа `61_237_000261_0-mixed.wav`

```bash
python3 train.py --config-path=src/configs --config-name=test.yaml +resume="/Users/leoproko/hse_ami/ml/dla2023/hw2_ss/ss_hw/checkpoint-epoch23.pth"
```
