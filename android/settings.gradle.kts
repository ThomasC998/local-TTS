// The phone half of Breeze TTS. Built on GitHub's runners rather than here --
// see .github/workflows/android.yml and the note in README-android.md.
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "BreezeRead"
include(":app")
