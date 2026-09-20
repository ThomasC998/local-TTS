plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.breezetts.read"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.breezetts.read"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            // Signed with the debug key: this is installed by hand on one
            // phone, not published, and a release key would be one more secret
            // to keep without buying anything.
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")

    // The media notification and its transport buttons come from here. Each
    // paragraph is one item in the player's playlist, so "next" is a paragraph.
    implementation("androidx.media3:media3-exoplayer:1.4.1")
    implementation("androidx.media3:media3-session:1.4.1")
    // ...and this is what makes ExoPlayer fetch through the pinned client
    // rather than through a connection that would trust any certificate.
    implementation("androidx.media3:media3-datasource-okhttp:1.4.1")

    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")

    // Pairing is a photograph of the Mac's screen; this reads it.
    implementation("com.journeyapps:zxing-android-embedded:4.3.0")
}
