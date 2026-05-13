# 🔊 Custom Electric Speaker Build & Acoustic Analysis

This repository contains the design, physical construction, and acoustic analysis of a hand-built electric speaker. The project was completed as part of the DT2212 Music Acoustics course at KTH Royal Institute of Technology.

📄 **Read the full academic report:** [`DT2212_Electric_Speaker_Project.pdf`](./DT2212_Electric_Speaker_Project.pdf)

The primary objective was to engineer a functional speaker from raw, repurposed materials, with a target frequency response between 100 Hz and 8000 Hz.

### Hardware Iteration & Construction
Building the speaker required balancing the total length of the voice coil wire and its resulting electrical resistance to maximize the Lorentz Force. 
* **Iteration 1:** The initial prototype utilized a thick conductive wire, a heavy doughnut-shaped magnet, rubber band suspension, and a plastic cup diaphragm. This design was highly inefficient, yielding a critically low electrical resistance of less than 1 ohm.
* **Iteration 2 & Final Build:** To optimize the magnetic gap and increase the wire length within the magnetic field, the thick wire was replaced with a 0.2mm gauge wire. The rubber band suspension was removed to reduce friction, and the plastic cup was replaced with a flat, rectangular coroplast diaphragm.

### Acoustic Measurements & Results
The speaker's frequency response was measured in a sound-treated room using a Behringer ECM8000 measurement microphone.
* **Testing Method:** We drove the speaker with a 60-second logarithmic sinusoidal chirp, sweeping continuously from 50 Hz to 20 kHz. 
* **Efficiency:** The custom speaker (34.8 Ω impedance) achieved an output of 42 dB(A) at 11.5V RMS. 
* **Acoustic Analysis:** The analysis of the audio sweep showed that the built speaker was highly inefficient at higher frequencies. Furthermore, the coroplast diaphragm had an uneven output across its surface because it lacked symmetry and had a unidirectional grain, which prevented it from acting as an ideal acoustic piston.

### Team
This was a collaborative group project done by:
* **Alex Ryström**
* **Robin Sachsenweger Ballantyne**
* **Sixten Bjuggren** 