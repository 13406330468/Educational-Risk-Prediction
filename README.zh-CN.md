# 校准迁移与更新：论文复现材料

论文：Calibration Transfer and Updating in Educational Risk Prediction: A Cross-Cohort Reliability Audit

## 上传 GitHub

解压外层交付压缩包，将其中的文件和文件夹上传到仓库根目录。根目录应直接看到 README.md、reproduce.py、Frozen_Results.zip、docs、paper、tests、work。

**上传时不要解压 Frozen_Results.zip，也不要解压 paper/LaTeX_Source.zip。** 前者解决原 aggregate_results.json 超过网页单文件限制的问题；其原始结果和 SHA-256 保持不变。解压后的大 JSON 只供审稿人在本地运行时使用，不需要再次上传。

## 审稿人本地核验

下载仓库后，在仓库根目录执行：

```bash
python -m zipfile -e Frozen_Results.zip .
python -m venv .venv
# 激活环境：PowerShell 使用 .venv/Scripts/Activate.ps1
# Linux/macOS 使用 source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py quick --run-dir ../calibration-quick
python -B tests/test_runtime_integrity.py -v
```

使用 Python 3.12。运行目录必须位于仓库之外，并使用新目录。quick 导出170项冻结汇总结果，核对评分恒等式，重新汇总三个人群的10万次课程聚类区间和288项更新比较，生成数字清单、表格和图；它不训练模型，不属于从原始数据独立复现。

## 从原始数据重算

原始 OULAD 和 XuetangX 数据需由使用者自行向数据发布方获取。数据来源链接及完整步骤见英文 README。

```bash
python reproduce.py prepare --run-dir ../calibration-full --dataset both --oulad-zip /path/OULAD.zip --xuetangx-zip /path/mooc-final.zip
python reproduce.py run --run-dir ../calibration-full --dataset both
python reproduce.py review3 --run-dir ../calibration-full --dataset xuetangx
```

最后一条命令执行基线、近期活跃、联合用户分配人群的两期和三期流程。也可只准备 XuetangX 后直接运行 review3。代码中 review3 等名称是保留的实现标识，不表示稿件经过期刊外审。完整研究证据由 run 与 review3 两部分覆盖。

作者此前已完成相关时间定义流程的原始日志重新准备和7,610次基础模型拟合。这是相同算法的再次执行，不是独立团队或不同实现的验证，也不意味着所有 OULAD 和历史扩展均已重新完整运行。本材料包更新稿件和分发方式，没有声称开展新的原始数据实验。

## 当前稿件及解释边界

- paper/main.pdf 与 paper/supplement.pdf 为最新稿件；paper/LaTeX_Source.zip 为配套主稿源码。现单位只保留两所学校：Ling Yu、Minghua Ding、Wei Sun、Yuguang Xing 对应山东外国语职业技术大学；Li Yu 对应山东水利职业学院，并担任通讯作者。
- 三个人群并非独立重复验证。近期活跃 Day 7 与基线重合；联合分配范围限于81个结构性候选课程对涉及的合格课程。
- 回顾性适配差距不保证未来更新获益。每个人群的六项完整预算 Full Platt 调整区间均跨零；限定条件下的收缩收益仍是探索性信号。
- 每个人群96项比较的调整不等于全论文统一校正。Bootstrap 条件于既有拟合，不包含全流程重拟合不确定性。原始标签生成窗口和实时日志到达时间仍未核实。
- 原始数据、学习者ID、个体预测、环境目录和私人作者资料均未重新分发。历史科学验证记录及冻结结果保留原标识，以便追溯。

MANIFEST.json 提供分发文件校验值；docs/VALIDATION.json 区分各项验证的范围。代码、数据和论文权利分别处理，既有第三方许可声明保持完整。此材料包不代表已经上传 GitHub；实际提交号应在上传后由仓库生成。
