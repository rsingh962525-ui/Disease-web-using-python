Option Explicit

Dim fso, shell, projectPath, pythonPath, modelPath, appPath, appUrl, command, attempt

Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

projectPath = fso.GetParentFolderName(WScript.ScriptFullName)
pythonPath = fso.BuildPath(projectPath, ".venv\Scripts\python.exe")
modelPath = fso.BuildPath(projectPath, "model\disease_model.joblib")
appPath = fso.BuildPath(projectPath, "app.py")
appUrl = "http://localhost:8501"

If Not fso.FileExists(pythonPath) Then
    MsgBox "App setup is missing. Follow the first-time setup instructions in README.md.", vbExclamation, "Disease Prediction System"
    WScript.Quit 1
End If

If Not fso.FileExists(modelPath) Then
    MsgBox "The trained model is missing. Follow the setup instructions in README.md first.", vbExclamation, "Disease Prediction System"
    WScript.Quit 1
End If

For attempt = 1 To 6
    If IsAppReady(appUrl) Then
        shell.Run appUrl, 1, False
        WScript.Quit 0
    End If
    WScript.Sleep 500
Next

shell.CurrentDirectory = projectPath
command = """" & pythonPath & """ -m streamlit run """ & appPath & """ --server.headless true --server.port 8501 --browser.gatherUsageStats false"
shell.Run command, 0, False

For attempt = 1 To 60
    WScript.Sleep 1000
    If IsAppReady(appUrl) Then
        shell.Run appUrl, 1, False
        WScript.Quit 0
    End If
Next

MsgBox "The app did not start. Check the setup instructions in README.md.", vbCritical, "Disease Prediction System"
WScript.Quit 1

Function IsAppReady(baseUrl)
    Dim request

    IsAppReady = False
    On Error Resume Next
    Err.Clear
    Set request = CreateObject("WinHttp.WinHttpRequest.5.1")
    If Err.Number = 0 Then
        request.SetTimeouts 500, 500, 500, 500
        request.Open "GET", baseUrl & "/_stcore/health", False
        request.Send
        If Err.Number = 0 Then
            IsAppReady = (request.Status = 200 And Trim(request.ResponseText) = "ok")
        End If
    End If
    On Error GoTo 0
End Function
