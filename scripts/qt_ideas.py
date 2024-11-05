
from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout,
                            QLabel, QPushButton, QScrollArea, QSizePolicy,
                            QStackedWidget, QTabWidget, QVBoxLayout, QWidget)
from qtpy.QtWidgets import QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMenu, QMenuBar, QPushButton, QScrollArea, QSizePolicy, QStackedWidget, QTabWidget, QVBoxLayout, QWidget
from qtpy.QtWidgets import QGraphicsEllipseItem, QGraphicsItem, QGraphicsPathItem, QGraphicsScene, QGraphicsView
from qtpy.QtCore import QPointF
from qtpy.QtWidgets import QApplication, QGraphicsEllipseItem, QGraphicsItem, QGraphicsPathItem, QGraphicsScene, QGraphicsView
from qtpy.QtGui import QPainterPath, QPainter, QPen

rad=5
class Node(QGraphicsEllipseItem):
    def __init__(self, path, index):
        super(Node, self).__init__(-rad, -rad, 2*rad, 2*rad)

        self.rad = rad
        self.path = path
        self.index = index

        self.setZValue(1)
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)
        self.setBrush(Qt.green)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            self.path.updateElement(self.index, value.toPoint())
        return QGraphicsEllipseItem.itemChange(self, change, value)


class Path(QGraphicsPathItem):
    def __init__(self, path, scene):
        super(Path, self).__init__(path)
        for i in range(path.elementCount()):
            node = Node(self, i)
            node.setPos(QPointF(path.elementAt(i)))
            scene.addItem(node)
        self.setPen(QPen(Qt.red, 1.75))        

    def updateElement(self, index, pos):
        path.setElementPositionAt(index, pos.x(), pos.y())
        self.setPath(path)


if __name__ == "__main__":

    app = QApplication([])

    path = QPainterPath()
    path.moveTo(0,0)
    path.cubicTo(-30, 70, 35, 115, 100, 100);
    path.lineTo(200, 100);
    path.cubicTo(200, 30, 150, -35, 60, -30);

    scene = QGraphicsScene()
    scene.addItem(Path(path, scene))

    view = QGraphicsView(scene)
    view.setRenderHint(QPainter.Antialiasing)
    view.resize(600, 400)
    view.show()
    app.exec_()


# class DragButton(QPushButton):

#     def mousePressEvent(self, event):
#         self.__mousePressPos = None
#         self.__mouseMovePos = None
#         if event.button() == Qt.LeftButton:
#             self.__mousePressPos = event.globalPos()
#             self.__mouseMovePos = event.globalPos()

#         super(DragButton, self).mousePressEvent(event)

#     def mouseMoveEvent(self, event):
#         if event.buttons() == Qt.LeftButton:
#             # adjust offset from clicked point to origin of widget
#             currPos = self.mapToGlobal(self.pos())
#             globalPos = event.globalPos()
#             diff = globalPos - self.__mouseMovePos
#             newPos = self.mapFromGlobal(currPos + diff)
#             self.move(newPos)

#             self.__mouseMovePos = globalPos

#         super(DragButton, self).mouseMoveEvent(event)

#     def mouseReleaseEvent(self, event):
#         if self.__mousePressPos is not None:
#             moved = event.globalPos() - self.__mousePressPos 
#             if moved.manhattanLength() > 3:
#                 event.ignore()
#                 return

#         super(DragButton, self).mouseReleaseEvent(event)

# def clicked():
#     print ("click as normal!")

# if __name__ == "__main__":
#     app = QApplication([])
#     w = QWidget()
#     w.resize(800,600)

#     button = DragButton("Drag", w)
#     button.clicked.connect(clicked)

#     w.show()
#     app.exec_()