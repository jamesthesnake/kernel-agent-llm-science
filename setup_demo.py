#!/usr/bin/env python3
"""
Setup script for Autonomous AI Research Demo

This script sets up the environment and runs the autonomous research pipeline demo
without requiring all heavy dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install basic requirements for the demo"""
    print("📦 Installing basic requirements...")

    basic_packages = [
        "pydantic>=2.5.0",
        "rich>=13.0.0",
        "numpy>=1.24.0"
    ]

    for package in basic_packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError:
            print(f"⚠️  Failed to install {package}, continuing...")

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking dependencies...")

    required = ['pydantic', 'json', 'pathlib', 'datetime']
    missing = []

    for pkg in required:
        try:
            __import__(pkg)
            print(f"✅ {pkg}")
        except ImportError:
            missing.append(pkg)
            print(f"❌ {pkg}")

    if missing:
        print(f"⚠️  Missing packages: {missing}")
        return False

    print("✅ All basic dependencies available!")
    return True

def run_demo():
    """Run the autonomous research demo"""
    print("\n🚀 Running Autonomous AI Research Demo...")

    # Import and run the demo
    try:
        from demo_full_autonomous_pipeline import run_autonomous_research_demo
        workspace, analysis, training, manuscript = run_autonomous_research_demo()

        print(f"\n✅ Demo completed successfully!")
        print(f"📁 Results saved in: {workspace}")

        return True

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main setup and demo execution"""
    print("🤖 AUTONOMOUS AI RESEARCH AGENT - SETUP & DEMO")
    print("=" * 60)

    # Install basic requirements
    try:
        install_requirements()
    except Exception as e:
        print(f"⚠️  Installation issues: {e}")

    # Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed")
        return False

    # Run demo
    success = run_demo()

    if success:
        print("\n🎉 SETUP AND DEMO COMPLETED!")
        print("🚀 Ready for Agents4Science submission!")
    else:
        print("\n❌ Setup or demo failed")

    return success

if __name__ == "__main__":
    main()