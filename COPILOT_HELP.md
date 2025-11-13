# GitHub Copilot Agent - What You Can Ask and Use It For

Welcome! This document explains what you can ask GitHub Copilot Agent and how to use it effectively in this KNMIproject repository.

## 🤖 What is GitHub Copilot Agent?

I'm an advanced AI coding assistant that can help you with various development tasks in this repository. I have access to your code, can make changes, run tests, and help with many development activities.

## ✨ What You Can Ask Me

### **Code Development**
- **Write new features**: "Add a function to calculate monthly averages"
- **Fix bugs**: "Fix the error in the temperature calculation"
- **Refactor code**: "Refactor the PlotFigure module to be more modular"
- **Add tests**: "Add unit tests for the Eobs.py module"
- **Improve performance**: "Optimize the data loading in Eobs.py"

### **Code Understanding**
- **Explain code**: "Explain how the plot_map function works"
- **Document code**: "Add docstrings to all functions in PlotFigure.py"
- **Review code**: "Review the Eobs.py file for potential issues"
- **Find patterns**: "Find all places where we use polyfit"

### **Data Analysis Tasks**
- **Add visualizations**: "Create a new plot type for precipitation trends"
- **Modify analysis**: "Change the trend calculation to use a different time period"
- **Add new variables**: "Add support for wind speed analysis"
- **Statistical operations**: "Add confidence intervals to the trend calculations"

### **Repository Management**
- **Setup and configuration**: "Add a requirements.txt file"
- **Documentation**: "Improve the README with usage examples"
- **Add examples**: "Create example scripts for common use cases"
- **Code organization**: "Reorganize the code into a proper Python package"

### **Debugging and Testing**
- **Debug issues**: "The plot is not showing correctly, help me debug"
- **Write tests**: "Add pytest tests for data processing functions"
- **Fix linting issues**: "Fix all PEP8 violations"
- **Performance profiling**: "Help me identify bottlenecks in data loading"

### **Integration and Automation**
- **Add CI/CD**: "Set up GitHub Actions for testing"
- **Add pre-commit hooks**: "Add linting checks before commits"
- **Automate tasks**: "Create a script to download and process data automatically"

## 🎯 Best Practices for Asking

### **Be Specific**
✅ Good: "Add error handling to the data loading function in Eobs.py when the file doesn't exist"
❌ Vague: "Make the code better"

### **Provide Context**
✅ Good: "I'm getting a KeyError when processing precipitation data. Can you fix this?"
❌ Unclear: "Fix the error"

### **Break Down Complex Tasks**
✅ Good: "First, add a function to validate input parameters. Then, add error messages."
❌ Too broad: "Make everything work perfectly"

## 📋 What This Repository Does

Based on the code analysis, this is a **KNMI Climate Data Analysis Project** that:

- **Processes European climate data** (E-OBS dataset)
- **Analyzes temperature and precipitation** trends over time
- **Creates map visualizations** of climate variables
- **Calculates linear trends** for regional climate analysis
- **Uses xarray and dask** for efficient data processing
- **Generates publication-quality maps** with Cartopy

### Current Capabilities:
- Temperature (Tg) analysis and visualization
- Precipitation (P) analysis and visualization
- Linear trend calculations over decades
- Regional data extraction
- Map plotting with custom colormaps
- Data downloading via CDS API

## 🚀 Example Questions You Can Ask

### For This Repository Specifically:

1. **"Add a function to calculate seasonal averages instead of annual"**
2. **"Create a new plot showing the difference between two time periods"**
3. **"Add command-line arguments to make the scripts more flexible"**
4. **"Write unit tests for the trend calculation"**
5. **"Add error handling for when the data files are missing"**
6. **"Create a Jupyter notebook example using these scripts"**
7. **"Add support for exporting results to CSV"**
8. **"Optimize the memory usage in data processing"**
9. **"Add a configuration file for user settings"**
10. **"Document the expected data format and directory structure"**

### General Development Tasks:

11. **"Explain the polyfit trend calculation method used"**
12. **"Add type hints to all functions"**
13. **"Create a setup.py for package installation"**
14. **"Add logging throughout the code"**
15. **"Create a comparison visualization between temperature and precipitation trends"**

## 🔧 How I Work

1. **I analyze your code** to understand the structure and context
2. **I make minimal, focused changes** - I won't rewrite everything
3. **I test my changes** when possible to ensure they work
4. **I provide explanations** of what I did and why
5. **I commit changes progressively** so you can review step-by-step

## 🛡️ What I Won't Do

- I won't access external services or APIs without your approval
- I won't share your code with third parties
- I won't introduce security vulnerabilities intentionally
- I won't delete working code unless necessary for the fix
- I won't access restricted files or system resources

## 💡 Tips for Best Results

1. **Start with clear questions** - I work best with specific requests
2. **Review my changes** - I'll report progress so you can verify
3. **Iterate if needed** - If something isn't quite right, ask me to adjust
4. **Ask for explanations** - I can explain any code or changes I make
5. **Think incrementally** - Complex tasks work best broken into steps

## 📚 Additional Help

- Ask me to explain any part of the existing code
- Request documentation for specific functions or modules
- Ask for best practices in Python, data analysis, or visualization
- Get help with git operations and repository management

## 🎓 Learning Resources

You can also ask me to:
- Explain climate data analysis concepts
- Teach you about xarray and dask usage
- Show best practices for scientific Python
- Guide you through Cartopy visualization techniques

---

**Remember**: I'm here to help you develop, understand, and improve your code. Don't hesitate to ask questions or request help with any development task!

Feel free to ask: **"What can you help me with for [specific task]?"**
