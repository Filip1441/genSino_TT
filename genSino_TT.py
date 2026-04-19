"""
GenSino-TT Main Entry Point
---------------------------
This script initializes and launches the GenSino-TT holographic tomography virtual laboratory.
It sets up the PySide6 application and displays the main user interface.
"""

import sys
from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow

def main():
    """Main function to launch the application."""
    app = QApplication(sys.argv)
    
    # Initialize and show the main window
    window = MainWindow()
    window.show()
    
    # Run the application event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()