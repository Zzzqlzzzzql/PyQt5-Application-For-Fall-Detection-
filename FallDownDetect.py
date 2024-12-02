import sys,os
from PyQt5.QtWidgets import QMainWindow,QApplication,QWidget
from Ui_FallDownDetect import Ui_MainWindow  #导入你写的界面类
from PyQt5 import QtCore,QtGui
 
 
class MyMainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self,parent =None):
        super(MyMainWindow,self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("FallDownDetect")
        self.setFixedSize(self.size())
        self.setWindowIcon(QtGui.QIcon('icons\main.png'))
        # 获取当前工作目录
        current_working_directory = os.getcwd()
        
        # 构建要检查/创建的文件夹路径
        folder_path = os.path.join(current_working_directory, 'saved')
        
        # 检查文件夹是否存在，如果不存在则创建
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    
 
if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())   