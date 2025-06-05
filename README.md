# Machine Vision Assignment: Sinogram Image Reconstruction

## 📘 Overview

This project demonstrates image reconstruction from a **parallel-projection sinogram** using Python. The process includes:
- Unfiltered Backprojection
- Ramp Filtered Backprojection
- Hamming-Windowed Ramp Filter
- Hann-Windowed Ramp Filter

Each RGB channel is reconstructed separately and combined into the final image.

**Contributors:**
- Luke Griffin  
- Patrick Crotty  
- Michael Cronin  
- Aaron Smith  
- Cullen Toal  

---

## 🧠 Background

Reconstruction from sinograms is a core concept in Computed Tomography (CT). This project builds upon:
- Radon Transform and Filtered Backprojection
- Frequency-domain filtering using Ramp, Hamming, and Hann windows
- Evaluation of visual clarity and artifact suppression in reconstruction

---

## 🏗️ Architecture & Workflow

### 1. Input Sinogram  
- Load the RGB sinogram  
- Extract metadata (e.g. aspect ratio)  

![image](https://github.com/user-attachments/assets/8e2c16fd-455a-4559-9790-cbf66645ad83)

**🖼️ Sinogram Image**

---

### 2. Preprocessing  
- Separate into R, G, B channels  
- Transform to frequency domain using FFT  

---

### 3. Filtering  
- Apply:
  - Ramp filter  
  - Hamming-windowed ramp filter  
  - Hann-windowed ramp filter  

![image](https://github.com/user-attachments/assets/36fb343c-cddb-4518-8abc-7622f6b9d0b8)

**🖼️ Hamming Window Plot**  

![image](https://github.com/user-attachments/assets/fb7ef5b5-ad9b-4dd2-84f6-15801a9a56f8)

**🖼️ Hann Window Plot**

---

### 4. Inverse Transformation  
- Convert filtered data back to the spatial domain using inverse FFT  

---

### 5. Backprojection  
- Reconstruct image by backprojecting each filtered projection at its corresponding angle  
- Accumulate results into a laminogram  

---

### 6. Post-Processing  
- Crop based on original aspect ratio  
- Normalize pixel values  
- Combine RGB channels into final image  

---

### 7. Output  
- Save reconstructed images for all filter types  
- Display side-by-side comparisons

![image](https://github.com/user-attachments/assets/9de5566b-e981-4259-bce3-e2a031e926e2)

**🖼️ No Filter Reconstruction**

![image](https://github.com/user-attachments/assets/4a317ffd-03d0-409d-8b28-5be53fa79100)

**🖼️ Ramp Filter Reconstruction**  

![image](https://github.com/user-attachments/assets/04c973d0-9f0d-42f0-ab22-b45e8f0b1fa2)

**🖼️ Hamming Ramp Filter Reconstruction**  

![image](https://github.com/user-attachments/assets/60374e9f-de43-4404-b699-0c991699e7ab)

**🖼️ Hann Ramp Filter Reconstruction**

---

## 📊 Results Summary

- **No Filter**: Blurry image with strong artifacts  
- **Ramp Filter**: Sharper image but some high-frequency noise  
- **Hamming**: Balanced sharpness and noise suppression  
- **Hann**: Smoother result, slight loss of detail  

---

## ✅ Conclusion

Filtering is essential in sinogram-based image reconstruction to suppress noise and improve clarity. The **Hamming window** provided the best compromise between detail and artifact reduction.

---

## 📎 References

- Zeng, G. L. (2014). *Model Based Filtered Backprojection Algorithm: A tutorial*.  
- Arai, Y. (2021). *Local cone beam CT: how did it all start?*  
- IAEA Human Health Campus. *3D Image Reconstruction*.  
- Wikipedia. *Radon Transform*  
- Wikipedia. *Backpropagation*  
- ScienceDirect. *Hamming Window*
