import numpy
import cv2
from tkinter import *
import tkinter.messagebox
from PIL import ImageTk,Image
import winsound

root = Tk()
root.geometry('500x640')
frame = Frame(root, relief=RIDGE, borderwidth=2)
frame.pack(fill=BOTH, expand=1)
root.title('Deteksi Masker')
frame.config(background='#CDE4EC')
root.resizable(False,False)
label = Label(frame, text="MASK DETEC\nAPPLICATION", bg='#CDE4EC' ,font=('Bodoni 24 bold'),pady=22)
label.pack(side=TOP)


filename = Image.open("C:\\Users\\Asus\\PyCharmProjects\\projekskripsi\\foto\\background.png")
ukuran = filename.resize((500,590),Image.ANTIALIAS)
newfilename = ImageTk.PhotoImage(ukuran)
background_label = Label(frame, image=newfilename, height=500, width=500)
background_label.pack()

def hel():
    help(cv2)

def Contri():
    tkinter.messagebox.showinfo("Author", "\nNauval Muhammad Dwiputra\n54417465\n\nContact:\nnauvalmd9921@gmail.com")

def anotherWin():
    tkinter.messagebox.showinfo("Application Version",
                                'version v1.0\n\n Made Using: \n* OpenCV\n* Numpy\n* Haarcascade Classifier\n* Tkinter\n* Python 3')


menu = Menu(root)
root.config(menu=menu)

subm1 = Menu(menu)
menu.add_cascade(label="About", menu=subm1)
subm1.add_command(label="Application", command=anotherWin)
subm1.add_command(label="Author", command=Contri)

#======================================== MENU EXIT ===============================================#

def exitt():
    exit()

#====================================== MENU DETEKSI ==============================================#

def webdet():

    # pengklasifikasian haarcascade (wajah, mata, mulut, badan bagian atas)

    face_cascade = cv2.CascadeClassifier('C:\\Users\\Asus\\PyCharmProjects\\projekskripsi\\data\\xml\\haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('data\\xml\\haarcascade_eye.xml')
    mouth_cascade = cv2.CascadeClassifier('C:\\Users\\Asus\\PyCharmProjects\\projekskripsi\\data\\xml\\haarcascade_mcs_mouth.xml')
    upper_body = cv2.CascadeClassifier('data\\xml\\haarcascade_upperbody.xml')

    # menyesuaikan nilai batas dalam kisaran 80 - 120 berdasarkan cahaya.
    bw_threshold = 105

    # DEKLARASI BEBERAPA VARIABLE

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (22, 22)
    warna_font_pakai_masker = (255, 255, 255)
    warna_font_tidak_pakai_masker = (0, 0, 255)
    threshold = 2
    font_skala = 1
    pakai_masker = " "
    tidak_pakai_masker = " "

    # Read video
    kamera = cv2.VideoCapture(1)

    while True:
        # ambil frame by frame
        ret, img = kamera.read()
        img = cv2.flip(img, 1)  # pengaturan posisi kamera horizontal + mirror (img,1) || kalau vertikal --> (img,0)

        # mengubah gambar menjadi gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # konversi gambar ke black and white
        (thresh, black_and_white) = cv2.threshold(gray, bw_threshold, 255, cv2.THRESH_BINARY)
        # cv2.imshow('black_and_white', black_and_white)

        # mendeteksi wajah
        wajah = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Face prediction for black and white
        faces_bw = face_cascade.detectMultiScale(black_and_white, 1.1, 4)

        if (len(wajah) == 0 and len(faces_bw) == 0):
            cv2.putText(img, pakai_masker, org, font, font_skala, warna_font_pakai_masker,
                        threshold, cv2.LINE_AA)
        elif (len(wajah) == 0 and len(faces_bw) == 1):
            # It has been observed that for white mask covering mouth, with gray image face prediction is not happening
            cv2.putText(img, pakai_masker, org, font, font_skala, warna_font_pakai_masker, threshold, cv2.LINE_AA)
        else:
            # membuat kotak pada wajah
            for (x, y, w, h) in wajah:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]

                # mendeteksi mulut
                mouth_rects = mouth_cascade.detectMultiScale(gray, 1.5, 5)

            # Face detected but Lips not detected which means person is wearing mask
            if (len(mouth_rects) == 0):
                cv2.putText(img, pakai_masker, org, font, font_skala, warna_font_pakai_masker, threshold, cv2.LINE_AA)
            else:
                for (mx, my, mw, mh) in mouth_rects:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_color = img[y:y + h, x:x + w]

                    if (y < my < y + h):
                        # Face and Lips are detected but lips coordinates are within face cordinates which `means lips prediction is true and
                        # person is not waring mask

                        cv2.putText(img, 'Tanpa Masker', (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.3, (255, 255, 255), 2)

                        #cv2.putText(img, 'no mask', org, font, font_skala, warna_font_tidak_pakai_masker, threshold,
                                    #cv2.LINE_AA)
                        #winsound.PlaySound('C:\\Users\\Asus\\PyCharmProjects\\projekskripsi\\peringatan.wav',winsound.SND_FILENAME)
                        cv2.rectangle(img, (mx, my), (mx + mh, my + mw), (0, 0, 255), 3)
                        break

        # Show frame with results
        cv2.putText(img, 'Jumlah Wajah Terdeteksi : ' + str(len(wajah)), (30, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    fontScale=1, color=(1, 0, 0))  # 1,(255,0,0),2)
        cv2.imshow('Deteksi Masker', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    # Release video
    kamera.release()
    cv2.destroyAllWindows()

#======================================== MENU TENTANG ===============================================#
def tentang():
    tentang_window = Toplevel(root)
    tentang_window.title("Tentang")
    tentang_window.geometry("600x570")
    tentang_window.configure(background="#CDE4EC")
    tentang_window.resizable(False,False)
    lbl_tentang = Label(tentang_window, text="\n\nAplikasi Deteksi Penggunaan Masker\n\n\n Dibuat oleh:\nNauval Muhammad Dwiputra\n54417465 ",
                        bg="#CDE4EC", fg="black", font="Verdana 10 bold")
    lbl_tentang.pack(pady=10)

     #FOTO
    fotoku = Image.open("C:\\Users\\Asus\\PyCharmProjects\\projekskripsi\\foto\\Kemeja.JPG")
    resized = fotoku.resize((120,150),Image.ANTIALIAS)
    newfoto = ImageTk.PhotoImage(resized)
    labelfoto = Label(tentang_window, image=newfoto, height=120, width=150)
    labelfoto.pack()

    lbl_tentang1 = Label(tentang_window,
                        text="\nAplikasi ini digunakan untuk mendeteksi penggunaan masker. \nJika seseorang menggunakan masker, "
                             "frame akan berwarna hijau. \nJika tidak menggunakan masker, frame berwarna merah dan bunyi peringatan.",
                        bg="#CDE4EC", fg="black", font="Verdana 10 bold")
    lbl_tentang1.pack()

    closebtn = Button(tentang_window, text="Close", command=lambda: tentang_window.destroy())
    closebtn.pack(pad=1)

#======================================== MENU MULAI RECORD ===============================================#
def webdetRec():
    # pengklasifikasian haarcascade (wajah, mata, mulut, badan bagian atas)

    face_cascade = cv2.CascadeClassifier('C:\\Users\\Asus\\PyCharmProjects\\projekskripsi\\data\\xml\\haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('data\\xml\\haarcascade_eye.xml')
    mouth_cascade = cv2.CascadeClassifier('C:\\Users\\Asus\\PyCharmProjects\\projekskripsi\\data\\xml\\haarcascade_mcs_mouth.xml')
    upper_body = cv2.CascadeClassifier('data\\xml\\haarcascade_upperbody.xml')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    rekam = cv2.VideoWriter('Record.avi', fourcc, 4.5, (640, 480))

    # menyesuaikan nilai batas dalam kisaran 80 - 120 berdasarkan cahaya.
    bw_threshold = 105

    # DEKLARASI BEBERAPA VARIABLE

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (22, 22)
    warna_font_pakai_masker = (255, 255, 255)
    warna_font_tidak_pakai_masker = (0, 0, 255)
    threshold = 2
    font_skala = 1
    pakai_masker = " "
    tidak_pakai_masker = " "

    # Read video
    kamera = cv2.VideoCapture(1)

    while True:
        # ambil frame by frame
        ret, img = kamera.read()
        img = cv2.flip(img, 1)  # pengaturan posisi kamera horizontal + mirror (img,1) || kalau vertikal --> (img,0)

        # mengubah gambar menjadi gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # konversi gambar ke black and white
        (thresh, black_and_white) = cv2.threshold(gray, bw_threshold, 255, cv2.THRESH_BINARY)
        # cv2.imshow('black_and_white', black_and_white)

        # mendeteksi wajah
        wajah = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Face prediction for black and white
        faces_bw = face_cascade.detectMultiScale(black_and_white, 1.1, 4)

        if (len(wajah) == 0 and len(faces_bw) == 0):
            cv2.putText(img, pakai_masker, org, font, font_skala, warna_font_pakai_masker,
                        threshold, cv2.LINE_AA)
        elif (len(wajah) == 0 and len(faces_bw) == 1):
            # It has been observed that for white mask covering mouth, with gray image face prediction is not happening
            cv2.putText(img, pakai_masker, org, font, font_skala, warna_font_pakai_masker, threshold, cv2.LINE_AA)
        else:
            # membuat kotak pada wajah
            for (x, y, w, h) in wajah:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]

                # mendeteksi mulut
                mouth_rects = mouth_cascade.detectMultiScale(gray, 1.5, 5)

            # Face detected but Lips not detected which means person is wearing mask
            if (len(mouth_rects) == 0):
                cv2.putText(img, pakai_masker, org, font, font_skala, warna_font_pakai_masker, threshold, cv2.LINE_AA)
            else:
                for (mx, my, mw, mh) in mouth_rects:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_color = img[y:y + h, x:x + w]

                    if (y < my < y + h):
                        # Face and Lips are detected but lips coordinates are within face cordinates which `means lips prediction is true and
                        # person is not waring mask

                        cv2.putText(img, 'Tanpa Masker', (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.3, (255, 255, 255),
                                    2)

                        # cv2.putText(img, 'no mask', org, font, font_skala, warna_font_tidak_pakai_masker, threshold,
                        # cv2.LINE_AA)
                        #winsound.PlaySound('C:\\Users\\Asus\\PyCharmProjects\\projekskripsi\\tanpa_ya.wav',winsound.SND_FILENAME)
                        cv2.rectangle(img, (mx, my), (mx + mh, my + mw), (0, 0, 255), 3)
                        break

        # Show frame with results
        cv2.putText(img, 'Jumlah Wajah Terdeteksi : ' + str(len(wajah)), (30, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    fontScale=1, color=(1, 0, 0))  # 1,(255,0,0),2)
        rekam.write(img)
        cv2.imshow('Deteksi Masker', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    # Release video
    rekam.release()
    kamera.release()
    cv2.destroyAllWindows()


button_mulai = Button(frame, padx=5, pady=5, width=39, bg='white', fg='black', relief=GROOVE, command=webdet,
              text='Open Cam & Detect\nTanpa Record', font=('helvetica 15 bold'))
button_mulai.place(x=5, y=220)

button_mulai_record = Button(frame, padx=5, pady=5, width=39, bg='white', fg='black', relief=GROOVE, command=webdetRec,
              text='Open Cam & Detect\nDengan Record', font=('helvetica 15 bold'))
button_mulai_record.place(x=5, y=310)

button_tentang = Button(frame, padx=5, pady=5, width=39, bg='white', fg='black', relief=GROOVE, command=tentang,
              text='Tentang', font=('helvetica 15 bold'))
button_tentang.place(x=5, y=400)


but5 = Button(frame, padx=5, pady=5, width=5, bg='white', fg='black', relief=GROOVE, text='EXIT', command=exitt,
              font=('helvetica 15 bold'))
but5.place(x=210, y=528)

root.mainloop()