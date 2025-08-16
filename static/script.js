document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('uploadForm');
    const queryForm = document.getElementById('queryForm');
    const responseOutput = document.getElementById('responseOutput');
    const uploadStatus = document.getElementById('uploadStatus');

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const fileInput = document.getElementById('reportFile');
        const file = fileInput.files[0];

        if (!file) {
            uploadStatus.textContent = 'Please select a file to upload.';
            return;
        }

        const formData = new FormData();
        formData.append('file', file);
        uploadStatus.textContent = 'Uploading and analyzing report...';

        try {
            const response = await fetch('/analyze-report', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            if (response.ok) {
                uploadStatus.textContent = result.message;
            } else {
                uploadStatus.textContent = `Error: ${result.detail}`;
            }
        } catch (error) {
            uploadStatus.textContent = 'An error occurred during upload.';
        }
    });

    queryForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const queryText = document.getElementById('queryText').value;

        if (!queryText) {
            responseOutput.textContent = 'Please enter a query.';
            return;
        }

        responseOutput.textContent = 'Getting insights...';

        try {
            const response = await fetch('/get-report-summary', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: queryText })
            });
            const result = await response.json();

            if (response.ok) {
                responseOutput.textContent = JSON.stringify(result, null, 2);
            } else {
                responseOutput.textContent = `Error: ${result.detail}`;
            }
        } catch (error) {
            responseOutput.textContent = 'An error occurred while fetching insights.';
        }
    });
});