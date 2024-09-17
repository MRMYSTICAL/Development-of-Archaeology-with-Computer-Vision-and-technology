import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import os
import webbrowser
import keyboard
import threading

class ScrollableFrame(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)

        canvas = tk.Canvas(self, bg='#222222')
        scrollbar = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas, bg='#222222')

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sparks AI Image Processing App")
        self.root.configure(bg='#222222')  # Dark theme background color

        self.input_image = None
        self.output_image = None
        self.selected_dataset = None  # Variable to store the selected dataset
        self.selected_model_files = False  # Variable to track if model files are selected for object detection

        # Tutorial completion flag
        self.tutorial_completed = False

        # Menu bar
        menubar = tk.Menu(self.root)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Save", command=self.save_image)
        file_menu.add_command(label="Exit", command=self.confirm_exit)
        menubar.add_cascade(label="File", menu=file_menu)

        # Shortcuts menu
        shortcuts_menu = tk.Menu(menubar, tearoff=0)
        shortcuts_menu.add_command(label="Manage Shortcuts", command=self.manage_shortcut_keybinds)
        menubar.add_cascade(label="Shortcuts", menu=shortcuts_menu)

        # Fullscreen menu
        self.fullscreen_var = tk.BooleanVar()
        self.fullscreen_var.set(False)
        menubar.add_checkbutton(label="Fullscreen", variable=self.fullscreen_var, command=self.toggle_fullscreen)

        # Tutorial menu
        menubar.add_command(label="Tutorial", command=self.launch_tutorial)

        # Credits menu
        credits_menu = tk.Menu(menubar, tearoff=0)
        credits_menu.add_command(label="Credits", command=self.show_credits)
        menubar.add_cascade(label="Credits", menu=credits_menu)

        self.root.config(menu=menubar)

        # Frames
        self.button_frame = tk.Frame(self.root, bg='#222222')
        self.button_frame.pack(side="top", padx=5, pady=5, fill="x")

        # Buttons
        self.load_button = tk.Button(self.button_frame, text="Load Image", command=self.load_image, bg='#444444', fg='white', padx=20, pady=10, relief=tk.RAISED)
        self.load_button.pack(side="left", padx=5, pady=5)

        self.compression_button = tk.Button(self.button_frame, text="Image Compression", command=self.compress_image, bg='#444444', fg='white', padx=20, pady=10, relief=tk.RAISED)
        self.compression_button.pack(side="left", padx=5, pady=5)

        self.resizing_button = tk.Button(self.button_frame, text="Image Resizing", command=self.resize_image_dialog, bg='#444444', fg='white', padx=20, pady=10, relief=tk.RAISED)
        self.resizing_button.pack(side="left", padx=5, pady=5)

        self.restore_button = tk.Button(self.button_frame, text="Restore Image", command=self.restore_image_popup, bg='#444444', fg='white', padx=20, pady=10, relief=tk.RAISED)
        self.restore_button.pack(side="left", padx=5, pady=5)

        self.detect_button = tk.Button(self.button_frame, text="Detect Objects", command=self.detect_objects, bg='#444444', fg='white', padx=20, pady=10, relief=tk.RAISED)
        self.detect_button.pack(side="left", padx=5, pady=5)
        
        self.enhance_button = tk.Button(self.button_frame, text="Enhance Image", command=self.enhance_image, bg='#444444', fg='white', padx=20, pady=10, relief=tk.RAISED)
        self.enhance_button.pack(side="left", padx=5, pady=5)

        # Import Pre-trained Model button
        self.import_model_button = tk.Button(self.button_frame, text="Import Pre-trained Model", command=self.import_pretrained_model, bg='#444444', fg='white', padx=20, pady=10, relief=tk.RAISED)
        self.import_model_button.pack(side="left", padx=5, pady=5)

        # Output frame
        self.output_scrollable_frame = ScrollableFrame(self.root)
        self.output_scrollable_frame.pack(side="top", padx=5, pady=5, fill="both", expand=True)

        # Save Output button
        self.save_output_button = tk.Button(self.root, text="Save Output", command=self.save_output, bg='#444444', fg='white', padx=20, pady=10, relief=tk.RAISED)
        self.save_output_button.pack(side="bottom", padx=5, pady=5)

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Variables for zooming
        self.input_zoom_factor = 1.0
        self.output_zoom_factor = 1.0

        # Thresholds
        self.thres = 0.45  # Threshold to detect object
        self.nms_threshold = 0.2

        # Model files and class names
        self.classFile = None
        self.configPath = None
        self.weightsPath = None
        self.classNames = None

        # Load model
        self.net = None

        # Shortcut keybinds dictionary
        self.shortcut_keybinds = {
            "load": "ctrl+o",
            "compress": "ctrl+c",
            "resize": "ctrl+r",
            "restore": "ctrl+e",
            "detect": "ctrl+d",
            "import_model": "ctrl+m",
            "enhance": "ctrl+n",
        }

        # Check if tutorial is completed
        if not self.tutorial_completed:
            self.launch_tutorial()

        # Register hotkeys
        for function_name, keybind in self.shortcut_keybinds.items():
            keyboard.add_hotkey(keybind, lambda func=function_name: self.execute_function(func))

        # Bind mouse wheel events for zooming
        self.root.bind("<MouseWheel>", lambda event: self.zoom_in(event, "input") if event.delta > 0 else self.zoom_out(event, "input"))
        self.root.bind("<Control-MouseWheel>", lambda event: self.zoom_in(event, "output") if event.delta > 0 else self.zoom_out(event, "output"))

    def toggle_fullscreen(self):
        self.root.attributes('-fullscreen', self.fullscreen_var.get())

    def manage_shortcut_keybinds(self):
        shortcut_keybinds_window = tk.Toplevel(self.root)
        shortcut_keybinds_window.title("Manage Shortcut Keybinds")
        shortcut_keybinds_window.geometry("400x300")
        shortcut_keybinds_window.configure(bg='#222222')

        shortcut_keybinds_label = tk.Label(shortcut_keybinds_window, text="Manage Shortcut Keybinds:", bg='#222222', fg='white', font=("Arial", 12, "bold"))
        shortcut_keybinds_label.pack(pady=10)

        # Function to change keybind for selected functionality
        def change_keybind(function_name, entry):
            new_keybind = entry.get()
            if new_keybind:
                keyboard.remove_hotkey(self.shortcut_keybinds[function_name])
                self.shortcut_keybinds[function_name] = new_keybind
                keyboard.add_hotkey(new_keybind, lambda func=function_name: self.execute_function(func))

        # Display all keybinds and entry widgets to change keybinds
        for function_name, keybind in self.shortcut_keybinds.items():
            frame = tk.Frame(shortcut_keybinds_window, bg='#222222')
            frame.pack(fill=tk.X, padx=10, pady=5)

            label = tk.Label(frame, text=function_name, bg='#222222', fg='white')
            label.pack(side=tk.LEFT)

            entry = tk.Entry(frame, bg='#444444', fg='white', relief=tk.FLAT)
            entry.insert(0, keybind)
            entry.pack(side=tk.LEFT, padx=(10, 0))

            change_button = tk.Button(frame, text="Change", command=lambda fn=function_name, ent=entry: change_keybind(fn, ent), bg='#444444', fg='white', relief=tk.RAISED)
            change_button.pack(side=tk.LEFT, padx=(10, 0))

    def execute_function(self, function_name):
        if function_name == "load":
            self.load_image()
        elif function_name == "compress":
            self.compress_image()
        elif function_name == "resize":
            self.resize_image_dialog()
        elif function_name == "restore":
            self.restore_image_popup()
        elif function_name == "enhance":
            self.enhance_image()
        elif function_name == "detect":
            if not self.selected_model_files:
                
                messagebox.showinfo("Select Model Files", "Please import pre-trained model files before proceeding with object detection.")
            else:
                self.detect_objects()
        elif function_name == "import_model":
            self.import_pretrained_model()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.input_image = cv2.imread(file_path)
            self.process_and_display_image()

    def display_image(self, image, frame, zoom_factor):
        aspect_ratio = image.shape[1] / image.shape[0]
        width = int(500 * zoom_factor)
        height = int(width / aspect_ratio)
        image = cv2.resize(image, (width, height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        label = tk.Label(frame, image=image, bg='#222222')
        label.image = image
        label.pack()

    def zoom_in(self, event, frame):
        if frame == "input":
            self.input_zoom_factor *= 1.1
            self.process_and_display_image()
        elif frame == "output":
            self.output_zoom_factor *= 1.1
            self.process_and_display_image()

    def zoom_out(self, event, frame):
        if frame == "input":
            self.input_zoom_factor /= 1.1
            self.process_and_display_image()
        elif frame == "output":
            self.output_zoom_factor /= 1.1
            self.process_and_display_image()

    def save_image(self, image=None):
        if image is not None:
            output_folder = filedialog.askdirectory()
            if output_folder:
                file_path = os.path.join(output_folder, "output_image.jpg")
                cv2.imwrite(file_path, image)
                return file_path

    def save_output(self):
        if self.output_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")])
            if file_path:
                cv2.imwrite(file_path, self.output_image)

    def compress_image(self):
        if self.input_image is not None:
            compression_quality = int(simpledialog.askstring("Compression Quality", "Enter compression quality (0-100):"))
            compressed_img = cv2.imencode('.jpg', self.input_image, [int(cv2.IMWRITE_JPEG_QUALITY), compression_quality])[1].tostring()
            nparr = np.frombuffer(compressed_img, np.uint8)
            self.output_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            self.process_and_display_image()

    def resize_image_dialog(self):
        if self.input_image is not None:
            width = simpledialog.askinteger("New Width", "Enter the new width (pixels):", parent=self.root)
            if width is not None:
                height = simpledialog.askinteger("New Height", "Enter the new height (pixels):", parent=self.root)
                if height is not None:
                    resized_img = cv2.resize(self.input_image, (width, height))
                    self.output_image = resized_img
                    self.process_and_display_image()

    def restore_image(self, threshold):
        if self.input_image is not None:
            # Convert to grayscale
            gray_image = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2GRAY)
            # Apply thresholding to detect scratches
            _, thresholded = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
            # Inpainting
            result_image = cv2.inpaint(self.input_image, thresholded, 3, cv2.INPAINT_TELEA)
            self.output_image = result_image
            self.process_and_display_image()

    def restore_image_popup(self):
        if self.input_image is not None:
            threshold = simpledialog.askinteger("Image Restoration", "Enter the threshold intensity (0-255):", minvalue=0, maxvalue=255)
            if threshold is not None:
                # Perform image restoration in a separate thread to avoid hanging the GUI
                threading.Thread(target=self.restore_image, args=(threshold,)).start()

    def detect_objects(self):
        if self.classFile is None or self.configPath is None or self.weightsPath is None:
            messagebox.showinfo("Select Model Files", "Please select class file, config file, and weights file for object detection.")
            return

        if self.input_image is not None:
            # Detect objects
            classIds, confs, bbox = self.net.detect(self.input_image, confThreshold=self.thres)
            if len(classIds) == 0:
                messagebox.showinfo("No Objects Detected", "No objects detected in the image. Please try again with a different pre-trained model.")
            else:
                # Find the index with the highest confidence score
                max_conf_idx = np.argmax(confs)

                # Get the bounding box with the highest confidence
                classId = classIds[max_conf_idx].flatten()
                confidence = confs[max_conf_idx].flatten()
                box = bbox[max_conf_idx]

                x, y, w, h = box
                cv2.rectangle(self.input_image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
                cv2.putText(self.input_image, self.classNames[classId[0] - 1].upper(), (x + 10, y + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                self.output_image = self.input_image
                self.process_and_display_image()

    def import_pretrained_model(self):
        # Change class file
        class_file = filedialog.askopenfilename(title="Select Class File", filetypes=[("name files", "*.names")])
        if class_file:
            self.classFile = class_file

        # Change config path
        config_path = filedialog.askopenfilename(title="Select Config Path", filetypes=[("Config files", "*.pbtxt")])
        if config_path:
            self.configPath = config_path

        # Change weights path
        weights_path = filedialog.askopenfilename(title="Select Weights Path", filetypes=[("Weight files", "*.pb")])
        if weights_path:
            self.weightsPath = weights_path

        if self.classFile is not None and self.configPath is not None and self.weightsPath is not None:
            # Read class names
            with open(self.classFile, 'rt') as f:
                self.classNames = f.read().rstrip('\n').split('\n')

            # Load model
            self.net = cv2.dnn_DetectionModel(self.weightsPath, self.configPath)
            self.net.setInputSize(320, 320)
            self.net.setInputScale(1.0 / 127.5)
            self.net.setInputMean((127.5, 127.5, 127.5))
            self.net.setInputSwapRB(True)
            self.selected_model_files = True

    def show_credits(self):
        credits_window = tk.Toplevel(self.root)
        credits_window.title("Credits")
        credits_window.geometry("700x500")
        credits_window.configure(bg='#222222', padx=10, pady=10)

        credits_label = tk.Label(credits_window, text="Credits: \n\n Project Developed Under the Guidences of DR.Mahantesh Kodabagi \n\nVarad Kulkarni - Data Collection, Application Coding\n\nVagish Kadakiya -   Software Testing and Research Paper Development\n\nSoham Maniar - Presentation and Software testing\n\n Organization:- D.Y Patil University, Ambi, Pune ", bg='#222222', fg='white', font=("Arial", 12, "bold"), justify=tk.CENTER)
        credits_label.pack()

    def process_and_display_image(self):
        # Clear previous output if exists
        for widget in self.output_scrollable_frame.scrollable_frame.winfo_children():
            widget.destroy()

        # Display input image preview
        if self.input_image is not None:
            self.display_image(self.input_image, self.output_scrollable_frame.scrollable_frame, self.input_zoom_factor)

        # Display output image if exists
        if self.output_image is not None:
            self.display_image(self.output_image, self.output_scrollable_frame.scrollable_frame, self.output_zoom_factor)

    def enhance_image(self):
        if self.input_image is not None:
            # Enhance the image (e.g., using filters, denoising, etc.)
            enhanced_image = cv2.bilateralFilter(self.input_image, 9, 75, 75)
            self.output_image = enhanced_image
            self.process_and_display_image() 
            
    def confirm_exit(self):
        if self.output_image is not None:
            result = messagebox.askyesno("Exit Confirmation", "Are you sure you want to quit without saving?")
            if result:
                self.root.destroy()
            else:
                self.save_output()
        else:
            self.root.destroy()

    def search_images(self):
        query = self.search_entry.get()
        # Open web browser with image search results
        if query:
            search_url = f"https://www.google.com/search?tbm=isch&q={query}"
            webbrowser.open(search_url)

    def show_coming_soon(self):
        messagebox.showinfo("Coming Soon", "This feature is coming soon!")

    def launch_tutorial(self):
        tutorial_window = tk.Toplevel(self.root)
        tutorial_window.title("Tutorial")
        tutorial_window.geometry("1000x700")
        tutorial_window.configure(bg='#222222')
        tutorial_window.attributes("-topmost", True)  # Make the tutorial window appear on top

        tutorial_label = tk.Label(tutorial_window, text="Welcome to Sparks AI Image Processing App Tutorial!\n\n"
                                                         "This tutorial will guide you through the functionalities of the application.\n\n"
                                                         "1. Load Image: Click this button to load an image for processing.\n"
                                                         "2. Image Compression: Compress the loaded image with desired quality.\n"
                                                         "3. Image Resizing: Resize the loaded image to desired dimensions.\n"
                                                         "4. Restore Image: Restore the image by removing scratches using thresholding(limit) and inpainting.\n."
                                                         "   (The user need 3 file .name file .config file and.weight file to use this feature)\n"
                                                         "   (Use the Import Pre-trained Model set button to import  this files)\n"
                                                         "5. Detect Objects: Detect objects in the loaded image using a pre-trained model.\n"
                                                         "6. Import Pre-trained Model: Import pre-trained model files for object detection.\n\n"
                                                         "You can also use shortcuts to perform these actions:\n\n"
                                                         "1) Ctrl+O: Load Image\n"
                                                         "2) Ctrl+C: Image Compression\n"
                                                         "3) Ctrl+R: Image Resizing\n"
                                                         "4) Ctrl+E: Restore Image\n"
                                                         "5) Ctrl+D: Detect Objects\n"
                                                         "6) Ctrl+M: Import Pre-trained Model\n"
                                                         "7) Use the Scrollwheel to Zoomin and Zoomout the Input\n "
                                                         "8) Use the Ctrl+Scrollwheel to Zoomin and Zoomout the Output\n\n"
                                                         "This App Can Give User the Flexiblity He/She Desired\n ."
                                                         "User can do multiple Operation on the Same Image\n\n "
                                                         "More features will be Added Later in the application\n\n"
                                                         "Feel free to explore and enjoy using the application!", bg='#222222', fg='white', font=("Arial", 12), justify=tk.LEFT)
        tutorial_label.pack(pady=10)

        got_it_button = tk.Button(tutorial_window, text="Got It!", command=tutorial_window.destroy, bg='#444444', fg='white', padx=20, pady=10, relief=tk.RAISED)
        got_it_button.pack(pady=10)

    # Register hotkeys
    # Mark tutorial as completed
        self.tutorial_completed = True


root = tk.Tk()
app = ImageProcessingApp(root)
root.mainloop()
