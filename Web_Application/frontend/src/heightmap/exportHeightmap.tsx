import { paramsType } from "../main";

export default function exportHandler(params: paramsType, heightmapsArray: Float64Array[], texture2d: HTMLImageElement) {
    const exportHeightmap = () => {
        // download satellite image as png
        var element = document.createElement('a');
        element.setAttribute('href', texture2d.src);
        element.setAttribute('download', 'satellite.png');
        document.body.appendChild(element);
        element.click();
        document.body.removeChild(element);

        // download heightmap as txt
        var element = document.createElement('a');
        element.setAttribute('href', 'data:text/plain;charset=utf-8, ' + encodeURIComponent(heightmapsArray[params.currentId].toString()));
        element.setAttribute('download', 'heightmap.txt');
        document.body.appendChild(element);
        element.click();
        document.body.removeChild(element);
    }
    const buttonExportHeightmap = document.getElementById('btn_export') as HTMLButtonElement;
    buttonExportHeightmap.addEventListener('click', exportHeightmap);
}

