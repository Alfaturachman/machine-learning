import pyautogui
import time

def auto_right_click(interval=1.0):
    """
    Fungsi ini akan melakukan klik kanan pada posisi mouse saat ini setiap interval detik.
    
    :param interval: Waktu dalam detik antara setiap klik kanan.
    """
    try:
        while True:
            pyautogui.rightClick()  # Klik kanan
            print("Klik kanan dilakukan.")
            time.sleep(interval)  # Tunggu sesuai interval waktu
    except KeyboardInterrupt:
        print("Program dihentikan oleh pengguna.")

if __name__ == "__main__":
    print("Program dimulai. Tekan Ctrl+C untuk menghentikan.")
    auto_right_click(interval=15.0)  # Misalnya, klik kanan setiap 2 detik
