export default function chooseDataset(params: any, btnCurrentDataset: HTMLButtonElement) {
    document.querySelectorAll('#datasets_list button').forEach(x => x.addEventListener('click', (event) => {
        const changeCurrentDataset = (datasetName: string, datasetBtn: HTMLButtonElement) => {
            params.currentDataset = datasetName;
            console.log(datasetBtn);
            btnCurrentDataset.style.backgroundColor = '';
            btnCurrentDataset = datasetBtn;
            btnCurrentDataset.style.backgroundColor = 'green';
        }
        switch(x.id) {
            case 'dataset_death_valley':
                changeCurrentDataset('death_valley', x as HTMLButtonElement);
                break;
            case 'dataset_laytonville':
                changeCurrentDataset('laytonville', x as HTMLButtonElement);
                break;
            case 'dataset_post_earthquake':
                changeCurrentDataset('post_earthquake', x as HTMLButtonElement);
                break;
            case 'dataset_mt_rainier':
                changeCurrentDataset('mt_rainier', x as HTMLButtonElement);
                break;
            case 'dataset_san_gabriel':
                changeCurrentDataset('san_gabriel', x as HTMLButtonElement);
                break;
        }
        console.log(params);
    }));
}