import { initializeApp } from 'firebase/app';
import { getAuth, GoogleAuthProvider, signInWithPopup, signOut } from 'firebase/auth';

const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY || "AIzaSyDGXADaGQX14BnXl8lGs4kLEPRGXnYYlQg",
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN || "apollo-7c7f6.firebaseapp.com",
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID || "apollo-7c7f6",
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET || "apollo-7c7f6.firebasestorage.app",
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID || "133497437042",
  appId: import.meta.env.VITE_FIREBASE_APP_ID || "1:133497437042:web:ab1aac1637849e7033fbeb"
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);

const provider = new GoogleAuthProvider();
provider.setCustomParameters({
  prompt: 'select_account'
});

export const signInWithGoogle = () => signInWithPopup(auth, provider);
export const signOutUser = () => signOut(auth);