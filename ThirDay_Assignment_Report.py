import pyautogui
import pyperclip
import time
import os
from datetime import datetime
from openpyxl import Workbook, load_workbook


# ==========================================================
# SETTINGS
# ==========================================================

# Folder where reports will be saved
REPORT_FOLDER = os.path.join(os.getcwd(), "NSE_Reports")

os.makedirs(REPORT_FOLDER, exist_ok=True)

# Today's date and time
now = datetime.now()

today_date = now.strftime("%Y-%m-%d")
current_datetime = now.strftime("%Y-%m-%d %H:%M:%S")

excel_filename = f"daily_report_{today_date}.xlsx"
excel_path = os.path.join(REPORT_FOLDER, excel_filename)

screenshot_filename = f"daily_report_{today_date}.png"
screenshot_path = os.path.join(REPORT_FOLDER, screenshot_filename)


# ==========================================================
# STEP 1 - OPEN CHROME
# ==========================================================

print("Step 1: Opening Chrome...")

pyautogui.hotkey("win", "r")
time.sleep(1)

pyautogui.write("chrome", interval=0.05)
pyautogui.press("enter")

time.sleep(4)


# ==========================================================
# STEP 1 - SELECT FIRST USER / PROFILE
# ==========================================================

print("Selecting first Chrome user/profile...")

# IMPORTANT:
# Adjust these coordinates according to your Chrome profile screen.
#
# Example coordinate:
# pyautogui.click(600, 400)

# Wait for Chrome profile selection screen
time.sleep(2)

# If Chrome opens directly into the first profile,
# you can comment the next line.
#
# Example:
# pyautogui.click(600, 400)

# For learning, we will simply press Enter.
pyautogui.press("enter")

time.sleep(3)


# ==========================================================
# STEP 2 - GO TO NSEINDIA.COM
# ==========================================================

print("Step 2: Opening NSE India...")

pyautogui.hotkey("ctrl", "l")
pyautogui.write("https://www.nseindia.com", interval=0.03)
pyautogui.press("enter")

time.sleep(8)


# ==========================================================
# STEP 3 - GO TO MARKET TURNOVER
# ==========================================================

print("Step 3: Going to Market Turnover...")

# ----------------------------------------------------------
# IMPORTANT:
#
# This part depends on the current NSE website layout.
#
# For learning PyAutoGUI, you can use image recognition
# or mouse coordinates.
#
# First try using keyboard navigation/search.
# ----------------------------------------------------------

# Open browser find
pyautogui.hotkey("ctrl", "f")
time.sleep(1)

pyautogui.write("Market Turnover", interval=0.05)
time.sleep(2)

pyautogui.press("esc")

# Press Enter if the found item is clickable
pyautogui.press("enter")

time.sleep(5)


# ==========================================================
# STEP 4 - COPY ALL INFORMATION FROM PAGE
# ==========================================================

print("Step 4: Copying page information...")

# Select all page content
pyautogui.hotkey("ctrl", "a")

time.sleep(1)

# Copy
pyautogui.hotkey("ctrl", "c")

time.sleep(2)

# Read clipboard
market_data = pyperclip.paste()

print("Data copied successfully.")

print("----------------------------------------")
print(market_data[:500])
print("----------------------------------------")


# ==========================================================
# STEP 5 - CREATE / OPEN EXCEL
# ==========================================================

print("Step 5: Opening Microsoft Excel...")

# Create a new Excel workbook directly using Python.
# This is more reliable than controlling the Excel GUI.

if os.path.exists(excel_path):

    workbook = load_workbook(excel_path)
    worksheet = workbook.active

else:

    workbook = Workbook()
    worksheet = workbook.active

    worksheet.title = "Daily NSE Report"

    # Header row
    worksheet["A1"] = "Date & Time"
    worksheet["B1"] = "NSE Market Turnover Data"


# ==========================================================
# STEP 6 - CREATE NEW ROW
# ==========================================================

print("Step 6: Adding today's data...")

# Find next empty row
next_row = worksheet.max_row + 1

# Date & Time
worksheet.cell(
    row=next_row,
    column=1,
    value=current_datetime
)

# NSE data
worksheet.cell(
    row=next_row,
    column=2,
    value=market_data
)


# Make column B wider
worksheet.column_dimensions["A"].width = 22
worksheet.column_dimensions["B"].width = 100


# Enable wrap text
worksheet.cell(
    row=next_row,
    column=2
).alignment = worksheet.cell(
    row=next_row,
    column=2
).alignment.copy(wrap_text=True)


# ==========================================================
# STEP 7 - SAVE EXCEL FILE
# ==========================================================

print("Step 7: Saving Excel report...")

workbook.save(excel_path)

print("Excel saved:")
print(excel_path)


# ==========================================================
# STEP 8 - OPEN EXCEL
# ==========================================================

print("Opening final Excel report...")

os.startfile(excel_path)

time.sleep(8)


# ==========================================================
# STEP 8 - TAKE SCREENSHOT
# ==========================================================

print("Step 8: Taking screenshot...")

screenshot = pyautogui.screenshot()

screenshot.save(screenshot_path)

print("Screenshot saved:")
print(screenshot_path)


# ==========================================================
# STEP 9 - CLOSE CHROME
# ==========================================================

print("Step 9: Closing Chrome...")

# Switch back to Chrome
pyautogui.hotkey("alt", "tab")

time.sleep(2)

# Close Chrome tab
pyautogui.hotkey("ctrl", "w")

time.sleep(2)


# ==========================================================
# FINISHED
# ==========================================================

print()
print("========================================")
print("      NSE DAILY REPORT COMPLETED")
print("========================================")
print()
print("Excel file:")
print(excel_path)
print()
print("Screenshot:")
print(screenshot_path)
print()
print("Automation completed successfully.")
