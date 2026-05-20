"""
模型对比实验主脚本
运行完整的对比实验流程
"""
import os
import subprocess

def main():
    print("="*100)
    print("🚀 启动模型对比实验")
    print("="*100)
    
    # Step 1: 训练所有模型
    print("\n📦 Step 1: 训练对比模型")
    print("-"*50)
    subprocess.run(['python', 'train_models.py'], cwd='/root/autodl-tmp/model_comparison')
    
    # Step 2: 评估所有模型
    print("\n📊 Step 2: 评估模型性能")
    print("-"*50)
    subprocess.run(['python', 'evaluate_models.py'], cwd='/root/autodl-tmp/model_comparison')
    
    print("\n" + "="*100)
    print("✅ 模型对比实验完成！")
    print("="*100)
    print("结果文件位置: /root/autodl-tmp/model_comparison/results/")

if __name__ == "__main__":
    main()
