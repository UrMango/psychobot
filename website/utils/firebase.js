import firebase from 'firebase/compat/app'
import 'firebase/compat/firestore'
import { getAuth } from 'firebase/auth';
// import firebase from 'firebase-admin';
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
// Here comes your firebase configration

// Your web app's Firebase configuration
const firebaseConfig = {
	apiKey: "AIzaSyCA4ctan_uDgAJXqlhweH0cG-lwcwJaQUA",
	authDomain: "psychobotai.firebaseapp.com",
	projectId: "psychobotai",
	storageBucket: "psychobotai.appspot.com",
	messagingSenderId: "445103675430",
	appId: "1:445103675430:web:cfe3b6fb5acb0db9b3aee7"
};
  
let app;


if(!firebase.apps.length) {
	app = firebase.initializeApp(firebaseConfig);
}

const firestore = firebase.firestore(app);
const auth = getAuth(app);

let token = "";

auth.onAuthStateChanged(async (user) => {
	if (user) {
		// console.log("PREV TOKEN:", token);
		token = await user.getIdToken();
		// console.log("NEW TOKEN:", token);
	} else {
		token = "";
	}
});

const getToken = () => token;

const onTokenChange = (func) => {
	auth.onAuthStateChanged(async (user) => {
		if (user) {
			// console.log("change");
			token = await user.getIdToken();
		} else {
			token = "";
		}
		// console.log(token);
		func(token);
	});
}

export {app, firestore, auth, getToken, token, onTokenChange};