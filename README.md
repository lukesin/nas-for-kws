# 基于神经结构搜索和量化的资源高效的DNN关键词定位[[arXiv]](https://arxiv.org/abs/2012.10138)
# [原作者地址](https://github.com/dapeter/nas-for-kws)
这个项目在 [Google Speech Commands](https://arxiv.org/abs/1804.03209) 数据集上使用 [ProxylessNAS](https://github.com/mit-han-lab/proxylessnas) [[arXiv]](https://arxiv.org/abs/1812.00332) [[Poster]](https://file.lzhu.me/projects/proxylessNAS/figures/ProxylessNAS_iclr_poster_final.pdf) 来实现资源高效的关键词检索CNN   
fork自 [nas-for-kws](https://github.com/dapeter/nas-for-kws) 针对单Python及Windows环境进行修改，跑了一个demo

### Prerequisites 先决条件
在Python3.9.5环境下使用了如下依赖 [requirements.txt](requirements.txt)

    pip install -r requirements.txt    

### Data 获取测试集
使用如下命令来获取 [Google Speech Commands](https://arxiv.org/abs/1804.03209) 测试集  
注意修改get_speech_commands.py中的绝对路径

    cd src/data/
    python get_speech_commands.py

## Running the code 运行测试
    通过下列两步来获取训练好的模型：  
    (1)执行NAS来获取一个较好的模型  
    (2)训练这个模型直至收敛  
### Efficient Architecture Search 高效搜索结构
####对下面的所有命令，都应该设置好输出路径
设置好BETA、OMEGA的值和输出路径，然后运行：

    python arch_search.py --path "output_path/" --dataset "speech_commands" --init_lr 0.2 --train_batch_size 100 --test_batch_size 100 --target_hardware "flops" --flops_ref_value 20e6 --n_worker 4 --arch_lr 4e-3 --grad_reg_loss_alpha 1 --grad_reg_loss_beta BETA --weight_bits 8 --width_mult OMEGA --n_mfcc 10
    python run_exp.py --path "output_path/learned_net" --train
本例暂时使用绝对路径：

    arch_search.py: --path "C:\Users\12994\PycharmProjects\nas-for-kws\src\search\output" --dataset "speech_commands" --init_lr 0.2 --train_batch_size 100 --test_batch_size 100 --target_hardware "flops" --flops_ref_value 20e6 --n_worker 4 --arch_lr 4e-3 --grad_reg_loss_alpha 1 --grad_reg_loss_beta 0.3 --weight_bits 8 --width_mult 1.0 --n_mfcc 10
    run_exp.py: --path "C:\Users\12994\PycharmProjects\nas-for-kws\src\search\output\learned_net" --train
### Weight quantization 权重量化
权重量化被作为一个后期处理环节，通过舍去一些网络参数来使用  
Weight quantization as a post-processing step by rounding parameters of a trained network is performed using
    
    python run_exp.py --path "output_path/learned_net" --quantize
使用 STE 的量化感知训练首先将“net.config”中所有层的“num_bits”更改为所需的位宽，然后运行  
Quantization aware training using the STE is performed by first changing "num_bits" of all layers in the "net.config" to the desired bit-width and then running

    python run_exp.py --path "output_path/learned_net" --train

### Varying Number of MFCC Features 不同数量的MFCC特征？

Set BETA and N_MFCC accordingly and run

    python arch_search.py --path "output_path/" --dataset "speech_commands" --init_lr 0.2 --train_batch_size 100 --test_batch_size 100 --target_hardware "flops" --flops_ref_value 20e6 --n_worker 4 --arch_lr 4e-3 --grad_reg_loss_alpha 1 --grad_reg_loss_beta BETA --weight_bits 8 --width_mult 1 --n_mfcc N_MFCC
    python run_exp.py --path "output_path/learned_net" --train

## Authors 作者

  - **Han Cai and Ligeng Zhu and Song Han** - Original code authors
  - **David Peter** - Updated code to perform NAS for KWS

<p><small>Template folder structure based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small><br>
<small>Readme based on <a target="_blank" href="https://github.com/PurpleBooth/a-good-readme-template">purple booths readme template</a>.</small></p>
