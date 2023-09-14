from tkinter import *
import numpy as np
from PIL import ImageGrab
from Prediction import predict

window = Tk()
window.title("Data Science's project")
l1 = Label()


def MyProject():
	global l1

	widget = cv
	# Seteamos las coordinadas del canvas
	x = window.winfo_rootx() + widget.winfo_x()
	y = window.winfo_rooty() + widget.winfo_y()
	x1 = x + widget.winfo_width()
	y1 = y + widget.winfo_height()

	# Capturamos la imagen y la convertimos en 28x28
	img = ImageGrab.grab().crop((x, y, x1, y1)).resize((28, 28))

	# Convertimos de RGB a escala de grises
	img = img.convert('L')

	# Extraer la matriz de píxeles de la imagen y convertirla en un vector de (1, 784)
	x = np.asarray(img)
	vec = np.zeros((1, 784))
	k = 0
	for i in range(28):
		for j in range(28):
			vec[0][k] = x[i][j]
			k += 1

	# Cargamos los thetas
	Theta1 = np.loadtxt('Theta1.txt')
	Theta2 = np.loadtxt('Theta2.txt')

	# Llamamos a la funcion "predecir"
	pred = predict(Theta1, Theta2, vec / 255)

	# Mostramos el resultado
	l1 = Label(window, text="Digit = " + str(pred[0]), font=('Calibri', 20))
	l1.place(x=230, y=500)


lastx, lasty = None, None


# Funcion para limpiar el canvas
def clear_widget():
	global cv, l1
	cv.delete("all")
	l1.destroy()


# Activamos el canvas
def event_activation(event):
	global lastx, lasty
	cv.bind('<B1-Motion>', draw_lines)
	lastx, lasty = event.x, event.y


# Funcion para dibujar el canvas
def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    if lastx and lasty:
        cv.create_line((lastx, lasty, x, y), width=30, fill='white', capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = x, y

# Crear un marco principal y centrarlo
main_frame = Frame(window)
main_frame.pack(expand=True, fill='both', padx=50, pady=50)  # Centrar y expandir

# Título
L1 = Label(main_frame, text="Number recognition", font=('Calibri', 25), fg="black")
L1.pack()  # Usar pack para centrar dentro del marco

# Configurar el lienzo
cv = Canvas(main_frame, width=500, height=350, bg='black')
cv.pack(pady=20)  # Agregar espacio debajo del lienzo

# Botón para limpiar el lienzo
b1 = Button(main_frame, text="1. Clear", font=('Calibri', 15), bg="white", fg="black", command=clear_widget)
b1.pack(side='left')  # Colocar a la izquierda dentro del marco

# Botón para predecir el dígito dibujado en el lienzo
b2 = Button(main_frame, text="2. Prediction", font=('Calibri', 15), bg="green", fg="white", command=MyProject)
b2.pack(side='right')  # Colocar a la derecha dentro del marco

cv.bind('<Button-1>', event_activation)
window.geometry("600x600")
window.mainloop()