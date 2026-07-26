$ErrorActionPreference = 'Stop'

$root = 'D:\TA Skripsi\aplikasi\HeightMeasurement'
$items = @(
    @{
        Input = Join-Path $root 'output\documents\LAPORAN_SKRIPSI_ROBUSTNESS_SITASI_SELARAS.docx'
        Output = Join-Path $root 'output\documents\rendered\LAPORAN_SKRIPSI_ROBUSTNESS_SITASI_SELARAS.pdf'
    },
    @{
        Input = Join-Path $root 'output\documents\JURNAL_ROBUSTNESS_SITASI_SELARAS.docx'
        Output = Join-Path $root 'output\documents\rendered\JURNAL_ROBUSTNESS_SITASI_SELARAS.pdf'
    }
)

New-Item -ItemType Directory -Force (Join-Path $root 'output\documents\rendered') | Out-Null

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$word.AutomationSecurity = 3
$word.Options.UpdateLinksAtOpen = $false
try {
    foreach ($item in $items) {
        Write-Output "Opening: $($item.Input)"
        $document = $word.Documents.Open($item.Input, $false, $true, $false, '', '', $false, '', '', 0, 0, $false, $true, 0, $true, '')
        try {
            Write-Output "Exporting: $($item.Output)"
            $document.ExportAsFixedFormat($item.Output, 17)
            Write-Output "Finished: $($item.Output)"
        }
        finally {
            $document.Close($false)
        }
    }
}
finally {
    $word.Quit()
}
