# FallDownDetect 项目 README
 
## 项目简介
 
`FallDownDetect` 是一个用于检测摔倒行为的项目。本文档将介绍项目GUI应用的运行环境、运行方法以及命令行指令。
 
## 运行环境
 
- **Python 版本**：请确保您的Python版本与主项目仓库中指定的版本兼容。
- **依赖包**：除了主项目仓库中列出的依赖项外，本项目还需要额外安装 `pyqt5` 包。您可以使用以下命令进行安装：
 
```bash
pip install pyqt5
```

请参照主项目仓库获取更详细的运行环境要求。

## 运行方法

要运行本项目，请使用Python执行 FallDownDetect.py 文件。在运行之前，请确保您已经阅读并理解了 property.txt 文件中的参数配置说明，以便根据实际需求进行调整。

```bash
python FallDownDetect.py
```

## 命令行指令

本项目支持多种命令行指令，用于执行原生项目的指令。这些指令的详细说明请参见 orders.txt 文件。
同时请注意检查`Ui_FallDownDetect_inferemote`两份代码中脚本路径，请根据实际本地路径进行修改。

## Windows/macOS

由于windows与mac系统中执行外部脚本的代码有所区别，因此请根据实际运行系统，在`FallDownDetect.py`中导入`Ui_FallDownDetect_inferemote_for_Mac`或者`Ui_FallDownDetect_Inferemote`