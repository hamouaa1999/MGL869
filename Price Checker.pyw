import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel, QFileDialog, QMessageBox, QTabWidget, QComboBox, QFrame
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QSettings, Qt
import sqlite3

class ImageFrame(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet("QLabel { background-color: white; }")
        self.setMinimumSize(200, 200)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

class DatabaseApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Price Checker")
        self.setMinimumSize(700, 400)  # Set minimum size

        self.layout = QHBoxLayout()
        self.setLayout(self.layout)

        # Create a tab widget
        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)

        # First tab for user input and data display
        self.tab1 = QWidget()
        self.tab_widget.addTab(self.tab1, "Data Display")
        self.tab1_layout = QVBoxLayout()
        self.tab1.setLayout(self.tab1_layout)

        # Sub-layout for search elements
        self.search_layout = QHBoxLayout()
        self.tab1_layout.addLayout(self.search_layout)

        # Widgets for user input
        self.search_by_label = QLabel("Search by:")
        self.search_layout.addWidget(self.search_by_label)

        self.search_by_combobox = QComboBox()
        self.search_by_combobox.addItems(["Name", "Barcode"])
        self.search_layout.addWidget(self.search_by_combobox)

        self.input_label = QLabel("Enter:")
        self.search_layout.addWidget(self.input_label)

        self.input_textbox = QLineEdit()
        self.search_layout.addWidget(self.input_textbox)

        # Button to retrieve data
        self.retrieve_button = QPushButton("Retrieve Data")
        self.retrieve_button.clicked.connect(self.retrieve_data)
        self.search_layout.addWidget(self.retrieve_button)

        # Connect returnPressed signal of input_textbox to retrieve_data method
        self.input_textbox.returnPressed.connect(self.retrieve_data)

        # Sub-layout for data display
        self.data_layout = QHBoxLayout()
        self.tab1_layout.addLayout(self.data_layout)

        # Sub-layout for image display
        self.image_layout = QVBoxLayout()
        self.data_layout.addLayout(self.image_layout)

        # Frame to hold image
        self.image_frame = ImageFrame()
        self.image_layout.addWidget(self.image_frame)

        # Sub-layout for ID, name, and price
        self.details_layout = QVBoxLayout()
        self.data_layout.addLayout(self.details_layout)

        # Labels to display retrieved data
        self.id_label = QLabel("ID:")
        self.details_layout.addWidget(self.id_label)

        self.name_label = QLabel("Barcode:")
        self.details_layout.addWidget(self.name_label)

        self.price_label = QLabel("Price:")
        self.details_layout.addWidget(self.price_label)

        # Second tab for open/close button
        self.tab2 = QWidget()
        self.tab_widget.addTab(self.tab2, "Connection Control")
        self.tab2_layout = QVBoxLayout()
        self.tab2.setLayout(self.tab2_layout)

        # Button to open/close connection
        self.conn_button = QPushButton("Open Connection")
        self.conn_button.clicked.connect(self.toggle_connection)
        self.tab2_layout.addWidget(self.conn_button)

        # Status label for connection status
        self.status_label = QLabel("Connection Closed")
        self.tab2_layout.addWidget(self.status_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Initialize database connection
        self.conn = None
        self.cur = None
        self.connection_open = False

        # Initialize QSettings
        self.settings = QSettings()

        # Connect dropdown index change signal
        self.search_by_combobox.currentIndexChanged.connect(self.update_labels)

    def connect_to_database(self, db_path):
        # Connect to the SQLite database
        try:
            self.conn = sqlite3.connect(db_path)
            self.cur = self.conn.cursor()
            self.connection_open = True
            self.status_label.setText("Connection Open")
            self.conn_button.setText("Close Connection")
            
            # Save last used database location
            self.settings.setValue("LastDatabaseLocation", db_path)
        except sqlite3.Error as e:
            QMessageBox.critical(self, "Error", f"Error connecting to database: {str(e)}")

    def close_database(self):
        # Close the database connection
        if self.conn:
            self.conn.close()
            self.connection_open = False
            self.status_label.setText("Connection Closed")
            self.conn_button.setText("Open Connection")

    def retrieve_data(self):
        if not self.connection_open:
            QMessageBox.information(self, "Information", "Database connection is closed. Open the connection first.")
            return

        search_by = self.search_by_combobox.currentText()
        search_term = self.input_textbox.text()

        if search_by == "Name":
            query = """
            SELECT product.id, product.name, product.price, barcode.value, product.image
            FROM product 
            JOIN barcode ON product.id = barcode.productId 
            WHERE product.name=?
            """
            id_label_text = "ID"
            name_label_text = "Barcode"
            price_label_text = "Price"
        else:
            query = """
            SELECT product.id, barcode.value, product.price, product.name, product.image
            FROM product 
            JOIN barcode ON product.id = barcode.productId 
            WHERE barcode.value=?
            """
            id_label_text = "ID"
            name_label_text = "Name"
            price_label_text = "Price"

        try:
            self.cur.execute(query, (search_term,))
            data = self.cur.fetchone()
            if data:
                self.id_label.setText(f"{id_label_text}: {data[0]}")
                self.name_label.setText(f"{name_label_text}: {data[3]}")
                self.price_label.setText(f"{price_label_text}: {data[2]}")

                # Display image if available
                image_data = data[4]
                if image_data:
                    image = QImage.fromData(image_data)
                    if not image.isNull():
                        pixmap = QPixmap.fromImage(image)
                        self.image_frame.setPixmap(pixmap)
                    else:
                        self.clear_image()  # Clear image if no data available
                else:
                    self.clear_image()  # Clear image if no data available
            else:
                self.clear_content()  # Clear all labels if no data found
                self.clear_image()    # Clear image if no data found
                QMessageBox.information(self, "Information", "No data found.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error retrieving data: {str(e)}")

    def clear_image(self):
        # Clear the image frame
        self.image_frame.clear()

    def toggle_connection(self):
        if self.connection_open:
            self.close_database()
        else:
            self.open_database()

    def open_database(self):
        # Retrieve last used database location
        last_db_location = self.settings.value("LastDatabaseLocation", "")
        
        # Open file dialog to choose the database file
        file_dialog = QFileDialog()
        if last_db_location:
            initial_path = last_db_location
        else:
            initial_path = ""
        db_path, _ = file_dialog.getOpenFileName(self, "Open Database File", initial_path, "SQLite Database Files (*.db *.sqlite)")
        if db_path:
            self.connect_to_database(db_path)
            
    def update_labels(self, index):
        self.clear_content()  # Clear all labels content on dropdown change
        self.clear_image()    # Clear image on dropdown change
        if index == 0:  # Name selected
            self.name_label.setText("Barcode:")
        else:  # Barcode selected
            self.name_label.setText("Name:")
    
    def clear_content(self):
        self.id_label.setText("ID:")
        self.name_label.setText("Name:")
        self.price_label.setText("Price:")

    def showEvent(self, event):
        # Override showEvent to center the window after it's shown
        self.center_window()

    def center_window(self):
        # Get the geometry of the primary screen
        screen_geometry = QApplication.primaryScreen().geometry()
        # Calculate the center of the screen
        x = (screen_geometry.width() - self.width()) / 2
        y = (screen_geometry.height() - self.height()) / 2
        # Move the window to the center
        self.move(x, y)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DatabaseApp()
    window.show()
    sys.exit(app.exec())
