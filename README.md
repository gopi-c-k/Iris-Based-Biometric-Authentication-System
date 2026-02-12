# 📚 SmartAttend – Offline WiFi-Based Secure Attendance System

SmartAttend is a full-stack attendance tracking system designed for secure, real-time classroom attendance without requiring continuous internet connectivity. It uses local WiFi discovery, biometric authentication, and real-time synchronization to ensure reliable and tamper-proof attendance recording.

This project consists of:

* 🎓 Faculty App (Flutter)
* 🎒 Student App (Flutter)
* 🌐 Backend API (Node.js + MongoDB)

---

## 🚀 Project Overview

SmartAttend enables teachers to start an attendance session locally over WiFi. Students automatically discover the session, authenticate using their device fingerprint, and securely mark attendance in real time.

The system works fully offline on a local network and later syncs data to the backend when internet connectivity becomes available. This architecture ensures both reliability and security in classroom environments.

The concept aligns with the offline, WiFi-based secure attendance workflow described in the SmartAttend invention proposal, where the teacher device acts as a local server and students connect through automatic discovery and biometric verification. 

---

## 🧩 Project Structure

```
SmartAttend/
│
├── faculty/   # Flutter app for faculty dashboard
├── student/   # Flutter app for student interactions
└── backend/   # Node.js + MongoDB REST API
```

### 📱 Faculty App

Flutter application used by teachers to:

* Start attendance sessions
* Generate classroom session codes
* View real-time attendance logs
* Generate attendance reports

### 🎒 Student App

Flutter application used by students to:

* Discover attendance session automatically
* Enter classroom code
* Verify identity using fingerprint
* Mark attendance securely

### 🌐 Backend

Node.js + MongoDB backend that:

* Handles authentication
* Stores attendance records
* Syncs offline attendance data
* Provides secure REST APIs

---

## ✨ Features

* 🔐 User Authentication (Faculty & Student)
* 📡 Offline WiFi-based Session Discovery
* 🧠 Mobile Fingerprint Authentication
* ⏱ Real-time Attendance Recording
* 🔄 Real-time Status Sync
* 🌍 Secure REST API with MongoDB
* 📊 Attendance Reports & Logs
* 📴 Works Fully Offline (Local Network Mode)
* ☁️ Later Online Sync Support

---

## 🏗️ System Architecture

1. Teacher starts an attendance session.
2. Session is broadcast over local WiFi.
3. Students automatically discover the session.
4. Students enter session code + verify fingerprint.
5. Attendance is recorded in real time on teacher device.
6. Data is later synced with the backend server.

This ensures only physically present students connected to the same local network can mark attendance, preventing proxy attendance.

---

## 🛠️ Tech Stack

### Mobile Apps

* Flutter (Faculty & Student)
* Dart
* Biometric Authentication APIs

### Backend

* Node.js
* Express.js
* MongoDB
* JWT Authentication

### Networking & Security

* Local WiFi-based communication
* mDNS / Service Discovery
* TCP socket-based attendance submission
* Fingerprint-based identity verification

---

## 📦 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/smartattend.git
cd smartattend
```

---

### 2️⃣ Backend Setup

```bash
cd backend
npm install
npm run dev
```

Create a `.env` file:

```
PORT=5000
MONGO_URI=your_mongodb_uri
JWT_SECRET=your_secret_key
```

---

### 3️⃣ Faculty App Setup

```bash
cd faculty
flutter pub get
flutter run
```

---

### 4️⃣ Student App Setup

```bash
cd student
flutter pub get
flutter run
```

---

## 🔒 Security Highlights

* Fingerprint authentication prevents proxy attendance
* Session-based classroom code validation
* Local network presence enforcement
* Real-time secure communication via sockets
* Offline-first architecture with later cloud sync

---

## 📊 Future Enhancements

* BLE-based proximity validation
* AI-based attendance anomaly detection
* Face recognition support (optional module)
* Analytics dashboard for attendance insights
* Multi-class & timetable integration

---

## 📄 License

This project is for academic and research purposes. Licensing can be updated based on deployment needs.

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub and share your feedback!
