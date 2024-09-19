# **Drone-Vyuh: Drone vs Bird Classification Using Micro-Doppler Radar Signatures**

## **Project Overview**

### **Description**
The increasing presence of unmanned aerial vehicles (UAVs) presents significant privacy and security risks. Drones, often confused with birds due to similar flight characteristics, require accurate identification to mitigate these threats, particularly in search and rescue operations, surveillance, and critical infrastructure protection.

This project, **Drone-Vyuh**, leverages radar-based micro-Doppler signatures to classify objects as either drones or birds. By analyzing distinct Doppler patterns from rotating drone propellers and bird wing beats, the system ensures reliable classification. The project also incorporates blockchain technology (Hyperledger Fabric) for secure data management and uses AWS SNS for real-time alerts based on security threats.

## **Problem Statement**

Unmanned aerial vehicles (UAVs), such as drones, are increasingly becoming a threat to privacy and security, requiring systems that provide situational awareness. One critical challenge is distinguishing between drones and birds in airspace, as they produce similar radar signatures due to moving parts (e.g., propellers and wings).

Micro-Doppler signatures captured from radar are used to classify these objects. However, traditional radar systems are insufficient for reliable classification, especially when distinguishing between drones and birds based on radar cross section (RCS) alone.

**Objective**: Use micro-Doppler signatures to accurately classify drones and birds for improved airspace security and protection of critical infrastructure.

## **Solution**

### **1. Radar Data Acquisition and Analysis**

**Micro-Doppler Signatures**:
- Micro-Doppler refers to frequency modulated components that are added to the Doppler signature due to the moving parts of an object, like a drone’s propeller or a bird’s wings.
- By analyzing these signatures using **Joint Time-Frequency Analysis**, we can extract patterns that differentiate drones from birds.

### **2. Classification System**

**Machine Learning Model**:
- We train a machine learning model to classify objects (drone or bird) based on their micro-Doppler signatures.
- Features like Doppler shifts from propeller blades or wing beats are key to classification accuracy.

**Input**: Micro-Doppler radar signatures.  
**Output**: Object classification – Drone or Bird.

### **3. Secure Storage Using Hyperledger Fabric**

To ensure the security and integrity of radar data, we use **Hyperledger Fabric**:
- **Chaincode**: Custom chaincode manages the radar data, handling classification metadata, and restricting access to authorized users only.
- **Private Data Collections**: Sensitive radar data is stored using private data collections, ensuring that only certain authorized members of the network can access the actual micro-Doppler data.
- **Customizable Consensus**: Hyperledger Fabric’s modular consensus algorithm guarantees scalability and ensures secure, immutable storage of data in the distributed ledger.

### **4. Augmented Reality Visualization**

To provide real-time visual feedback, we integrate **Augmented Reality (AR)**:
- AR overlays real-time radar data and classification results onto an interactive, immersive interface.
- This allows operators to view the movement and classification of objects (drones or birds) in real-time, enhancing situational awareness and decision-making.

### **5. Real-Time IoT Alerts Using AWS SNS**

We incorporate **AWS Simple Notification Service (SNS)** for real-time alerts:
- **AWS Lambda** processes classification results from the ML model and triggers alerts when a drone is detected.
- AWS SNS sends notifications (email, SMS, or IoT alerts) to authorized personnel, enabling prompt response to potential security threats.

---

## **Implementation Details**

### **Step 1: Radar Data Collection and Preprocessing**
- **Data Acquisition**: Use an FMCW radar system to collect micro-Doppler data from drones and birds.
- **Preprocessing**: Perform Joint Time-Frequency Analysis on the raw radar data to extract useful features for classification.

### **Step 2: Machine Learning Classification Model**
- **Model Training**: Train a machine learning model (e.g., SVM or neural network) using labeled micro-Doppler data (drone or bird).
- **Evaluation**: Test and validate the model to ensure accuracy in differentiating between drones and birds.

### **Step 3: Blockchain Integration with Hyperledger Fabric**
- **Setup Hyperledger Fabric**: Deploy a permissioned blockchain network with private data collections to store radar data securely.
- **Chaincode Development**: Write smart contracts (chaincode) to handle access control and manage drone classification metadata.
  - Use consensus mechanisms to ensure data immutability and security.
  
### **Step 4: Augmented Reality Visualization**
- **AR Development**: Use an AR framework (such as Unity or ARKit) to build a real-time visual interface that overlays radar data and classification results on an interactive display.
- **Integration**: Connect the AR interface to the backend radar data and classification system to show live updates.

### **Step 5: Real-Time IoT Alerts with AWS SNS**
- **AWS Lambda**: Configure AWS Lambda to automatically process classification outputs.
- **AWS SNS**: Set up AWS SNS to send real-time notifications (SMS, email) when a drone is detected, allowing immediate response to potential threats.

---

## **Future Enhancements**
- **Expand Classification to Multiple Objects**: Extend the system to classify additional airborne objects, such as helicopters or larger aircraft.
- **Enhanced Privacy with Homomorphic Encryption**: Integrate homomorphic encryption for even greater privacy in radar data sharing.

---

## **Contributors**
- **Your Name**: Project lead and developer.
- **Team Members**: Collaborators, if any.

---

## **License**
This project is licensed under the MIT License.

---

## **Contact**
For more information, contact **sagnikdas5432@gmail.com**.
